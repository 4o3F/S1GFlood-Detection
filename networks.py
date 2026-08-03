import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from NormalCell import Mlp
from timm.models.layers import trunc_normal_
import math
import functools
from einops import rearrange
from resnet import ResNet
from base_model import ViTAE_Window_NoShift_basic
from swin_transformer import swin


class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels), nn.ReLU(), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


def get_scheduler(optimizer, args):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3

        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) 


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids) 
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Backbone(torch.nn.Module):
    def __init__(self, args, input_nc, output_nc, resnet_stages_num=5, output_sigmoid=False, if_upsample_2x=True):
        super(Backbone, self).__init__()
        if args.backbone == 'resnet':
            self.backbone = ResNet(args)
            filters0 = [256, 512, 1024, 2048]

        elif args.backbone == 'swin':
            self.backbone = swin(args, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.3, attn_drop_rate=0.,
                                 drop_path_rate=0.3, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), use_checkpoint=False,
                                 frozen_stages=-1, norm_eval=False)

            filters0 = [96, 192, 384, 768]

        elif args.backbone == 'vitae':
            self.backbone = ViTAE_Window_NoShift_basic(args, RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], 
                                                       NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], stages=4, 
                                                       embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                                                       NC_depth=[2, 2, 8, 2], NC_heads=[1, 2, 4, 8], RC_heads=[1, 1, 2, 4], mlp_ratio=4., NC_group=[1, 32, 64, 128], 
                                                       RC_group=[1, 16, 32, 64], img_size=1024, window_size=7, drop_path_rate=0.3, frozen_stages=-1, norm_eval=False)

            filters0 = [64, 128, 256, 512]
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.resnet_stages_num = resnet_stages_num
        self.if_upsample_2x = if_upsample_2x
        layers = filters0[-2] 
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        x = self.backbone(x)
        x0, x1, x2, x3 = x

        if self.if_upsample_2x:
            x = self.upsamplex2(x2)
        else:
            x = x2
        x = self.conv_pred(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class I2T(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b c h w -> b (h w) c')
    

class AttentionWrapper(nn.Module):
    def __init__(self, dim, num_heads):
        self.attn = Attention(dim=dim, heads=num_heads)
        super().__init__()
        self.num_heads = num_heads 
        
    def forward(self, x, mask=None):
        return self.attn(x, mask)
        

class CrossTemporalChangeAttention(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = token_dim
        self.proj_q = nn.Linear(token_dim, token_dim)
        self.proj_k = nn.Linear(token_dim, token_dim)
        self.proj_v = nn.Linear(token_dim, token_dim)
        mlp_hidden_dim = int(token_dim * 4) 
        self.MLP = Mlp(in_features=token_dim, hidden_features=mlp_hidden_dim) 
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, R_pre, R_post):
        Q_pre, K_pre, V_pre = self.proj_q(R_pre), self.proj_k(R_post), self.proj_v(R_post)
        Q_minus_K_pre = Q_pre - K_pre
        Q_norm_pre = F.normalize(Q_minus_K_pre, p=2, dim=-1)
        K_norm_pre = F.normalize(K_pre, p=2, dim=-1) 
        attn_weights_pre = torch.softmax(torch.matmul(Q_norm_pre, K_norm_pre.transpose(-2, -1)) / (self.token_dim ** 0.5), dim=-1)
        attn_output_pre = torch.matmul(attn_weights_pre, V_pre)
        F_c_pre = self.MLP(self.norm1(attn_output_pre)) + attn_output_pre
        
        Q_post, K_post, V_post = self.proj_q(R_post), self.proj_k(R_pre), self.proj_v(R_pre)
        Q_minus_K_post = Q_post - K_post
        Q_norm_post = F.normalize(Q_minus_K_post, p=2, dim=-1)
        K_norm_post = F.normalize(K_post, p=2, dim=-1)
        attn_weights_post = torch.softmax(torch.matmul(Q_norm_post, K_norm_post.transpose(-2, -1)) / (self.token_dim ** 0.5), dim=-1)
        attn_output_post = torch.matmul(attn_weights_post, V_post)
        F_c_post = self.MLP(self.norm2(attn_output_post)) + attn_output_post
        
        return F_c_pre, F_c_post


class TemporalAwareChangeEnhancement(nn.Module):
    def __init__(self, token_dim, H, W, is_last_stage=False):
        super().__init__()
        self.is_last_stage = is_last_stage
        self.H, self.W = H, W
        mlp_hidden_dim = int(token_dim * 4)

        self.PCM = nn.Sequential(
            nn.Conv2d(token_dim, mlp_hidden_dim, 3, 1, 1, 1, 1),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(mlp_hidden_dim, token_dim, 3, 1, 1, 1, 1),
            nn.BatchNorm2d(token_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, 3, 1, 1, 1, 1),
        )

        self.i2t = I2T()
        self.MHA = Attention(dim=token_dim, heads=4)
        self.MHA.num_heads = 4
        self.FFN = Mlp(in_features=token_dim, hidden_features=mlp_hidden_dim)
        self.norm1 = nn.LayerNorm(token_dim)

        if is_last_stage:
            self.class_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            trunc_normal_(self.class_token, std=.02)
            # Register Q/K/V projections so optimizer and checkpoints track them.
            self.proj_q = nn.Linear(token_dim, token_dim, bias=False)
            self.proj_k = nn.Linear(token_dim, token_dim, bias=False)
            self.proj_v = nn.Linear(token_dim, token_dim, bias=False)

    def forward(self, R_i, F_c_i):
        H, W = self.H, self.W
        B, N_c, C = F_c_i.shape
        B, N_r, C = R_i.shape

        H_short, W_short = int(math.sqrt(N_c)), int(math.sqrt(N_c))
        F_c_map_short = F_c_i.transpose(1, 2).reshape(B, C, H_short, W_short)
        F_c_map = F.interpolate(F_c_map_short, size=(H, W), mode='bilinear', align_corners=False)

        F_pcm = self.PCM(F_c_map)
        token_pcm = self.i2t(F_pcm)

        t_sem = None
        if self.is_last_stage:
            cls_tokens = self.class_token.expand(B, -1, -1)

            Q_flat = self.proj_q(R_i)
            K_flat = self.proj_k(F_c_i)
            V_flat = self.proj_v(F_c_i)
            Q_cls = self.proj_q(cls_tokens)
            Q_flat = torch.cat([Q_cls, Q_flat], dim=1)

            attn = torch.matmul(Q_flat, K_flat.transpose(-2, -1)) / (C ** 0.5)
            attn = attn.softmax(dim=-1)
            attn_output = torch.matmul(attn, V_flat)

            t_sem = attn_output[:, 0]
            attn_output = attn_output[:, 1:]
        else:
            attn = torch.matmul(R_i, F_c_i.transpose(-2, -1)) / (C ** 0.5)
            attn = attn.softmax(dim=-1)
            attn_output = torch.matmul(attn, F_c_i)

        F_tace = attn_output + token_pcm
        F_e_token = self.FFN(self.norm1(F_tace)) + F_tace

        F_e_map = F_e_token.transpose(1, 2).reshape(B, C, H, W)
        return F_e_map, t_sem
    

class TemporalDifferentialFusion(nn.Module):
    def __init__(self, token_dim, output_nc):
        super().__init__()
        self.token_dim = token_dim
        self.conv_blocks = nn.ModuleList([ConvBNReLU(token_dim * 2, token_dim, kernel_size=1) for _ in range(4)])
        self.fuse_conv = ConvBNReLU(token_dim * 4, token_dim, kernel_size=1)
        self.mlp_sem = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))
        
        self.pred_head = nn.Sequential(
            nn.ConvTranspose2d(token_dim, token_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(token_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(token_dim // 2, token_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(token_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(token_dim // 4, output_nc, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, F_e_pre, F_e_post, t_sem):
        if not isinstance(F_e_pre, list): 
            F_e_pre = [F_e_pre] * 4 
            F_e_post = [F_e_post] * 4

        temporal_differential_features = []
        B, C, H, W = F_e_pre[0].shape
        
        for i in range(4):
            concat_features = torch.cat([F_e_pre[i], F_e_post[i]], dim=1)
            tdf_conv = self.conv_blocks[i](concat_features)
            temporal_differential_features.append(tdf_conv)

        fused_features = temporal_differential_features[0]
        for i in range(1, 4):
            upsampled_feature = F.interpolate(temporal_differential_features[i], size=(H, W), mode='bilinear', align_corners=False)
            fused_features = torch.cat([fused_features, upsampled_feature], dim=1)
            
        F_fuse = self.fuse_conv(fused_features)
        
        t_sem_proj = self.mlp_sem(t_sem)
        t_sem_proj = t_sem_proj.view(B, C, 1, 1).expand_as(F_fuse)
        F_enhanced = F_fuse + t_sem_proj
        
        M = self.pred_head(F_enhanced)
        return M


class WaterSegmentationHead(nn.Module):
    def __init__(self, in_channels=32, hidden_channels=16, output_nc=2):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, output_nc, kernel_size=1),
        )

    def forward(self, feature_map, output_size):
        logits = self.refine(feature_map)
        return F.interpolate(
            logits,
            size=output_size,
            mode='bilinear',
            align_corners=False,
        )


class DAMNet_New(Backbone):
    def __init__(self, args, input_nc, output_nc, with_pos, resnet_stages_num=5, token_len=4, token_trans=True, enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64, tokenizer=True, if_upsample_2x=True, pool_mode='max', pool_size=2, backbone='vitae',
                 decoder_softmax=True, with_decoder_pos=None, with_decoder=True):
        
        self.input_nc = input_nc 
        super(DAMNet_New, self).__init__(args, input_nc, output_nc, resnet_stages_num=resnet_stages_num, if_upsample_2x=if_upsample_2x)

        self.token_len = token_len
        dim = 32
        mlp_dim = 2*dim
        
        self.conv_a = nn.Conv2d(dim, self.token_len, kernel_size=1, padding=0, bias=False)
        self.tokenizer = tokenizer
        
        self.token_trans = token_trans
        self.with_decoder = with_decoder

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, dim))
        
        decoder_pos_size = 256//8
        H_map, W_map = decoder_pos_size, decoder_pos_size
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, dim, H_map, W_map))
            
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8, dim_head=self.dim_head, mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth, heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0, softmax=decoder_softmax)
        self.CTCA = CrossTemporalChangeAttention(token_dim=dim)
        self.TACE_pre = TemporalAwareChangeEnhancement(token_dim=dim, H=H_map, W=W_map, is_last_stage=True)
        self.TACE_post = TemporalAwareChangeEnhancement(token_dim=dim, H=H_map, W=W_map, is_last_stage=True)
        self.TDF = TemporalDifferentialFusion(token_dim=dim, output_nc=output_nc)
        self.water_head = WaterSegmentationHead(
            in_channels=dim,
            hidden_channels=dim // 2,
            output_nc=2,
        )

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens
    
    
    def _forward_reshape_tokens(self, x):
        if self.pool_mode=='max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode=='ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x_token = rearrange(x, 'b c h w -> b (h w) c')
        x_token = self.transformer_decoder(x_token, m)
        x_map = rearrange(x_token, 'b (h w) c -> b c h w', h=h)
        return x_map, x_token


    def forward(self, x1, x2, return_aux=False):
        output_size_a = x1.shape[-2:]
        output_size_b = x2.shape[-2:]
        x1_map = self.forward_single(x1)
        x2_map = self.forward_single(x2)

        if return_aux:
            if not hasattr(self, 'water_head'):
                raise RuntimeError(
                    'This checkpoint does not contain the auxiliary water head. '
                    'Use default change inference or train a new checkpoint.'
                )
            water_a_logits = self.water_head(x1_map, output_size_a)
            water_b_logits = self.water_head(x2_map, output_size_b)

        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1_map)
            token2 = self._forward_semantic_tokens(x2_map)
        else:
            token1 = self._forward_reshape_tokens(x1_map)
            token2 = self._forward_reshape_tokens(x2_map)

        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            R = self._forward_transformer(self.tokens_)
            R_pre, R_post = R.chunk(2, dim=1)
        else:
            R_pre, R_post = token1, token2
        
        F_c_pre, F_c_post = self.CTCA(R_pre, R_post)
        
        _, R_dec_pre = self._forward_transformer_decoder(x1_map, R_pre)
        _, R_dec_post = self._forward_transformer_decoder(x2_map, R_post)

        F_e_pre_map, t_sem_pre = self.TACE_pre(R_dec_pre, F_c_pre)
        F_e_post_map, t_sem_post = self.TACE_post(R_dec_post, F_c_post)
        
        t_sem = (t_sem_pre + t_sem_post) / 2
        x = self.TDF(F_e_pre_map, F_e_post_map, t_sem)
        if self.output_sigmoid:
            x = self.sigmoid(x)

        if not return_aux:
            return x
        return {
            'change_logits': x,
            'water_a_logits': water_a_logits,
            'water_b_logits': water_b_logits,
        }