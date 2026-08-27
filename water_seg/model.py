import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from swin_transformer import swin


_TIMM_SWIN_TINY = 'swin_tiny_patch4_window7_224'
_MIN_MATCHED_PRETRAINED_TENSORS = 100
_PATCH_EMBED_WEIGHT = 'patch_embed.proj.weight'
_POLARIZATIONS = ('VV', 'VH')


def _is_ignored_pretrained_key(name):
    if name == 'head' or name.startswith('head.'):
        return True
    return name.endswith('attn_mask')


def _load_timm_swin_tiny_weights(encoder):
    import timm
    from timm.models.helpers import adapt_input_conv

    source = timm.create_model(_TIMM_SWIN_TINY, pretrained=True, in_chans=3)
    source_state = source.state_dict()
    dest_state = encoder.state_dict()
    matched = {}
    for name, tensor in source_state.items():
        if _is_ignored_pretrained_key(name):
            continue
        destination_name = name
        if name.startswith('norm.'):
            destination_name = name.replace('norm.', 'norm3.', 1)
        dest = dest_state.get(destination_name)
        if dest is None:
            continue
        if name == _PATCH_EMBED_WEIGHT:
            tensor = adapt_input_conv(len(_POLARIZATIONS), tensor)
        if dest.shape == tensor.shape:
            matched[destination_name] = tensor
    if len(matched) < _MIN_MATCHED_PRETRAINED_TENSORS:
        raise RuntimeError(
            'failed to load ImageNet Swin-T encoder weights: '
            f'matched {len(matched)} tensors, expected at least '
            f'{_MIN_MATCHED_PRETRAINED_TENSORS}'
        )
    encoder.load_state_dict(matched, strict=False)


class SwinTinyEncoder(swin):
    def __init__(self, imagenet_pretrained=False):
        super().__init__(
            None,
            in_chans=len(_POLARIZATIONS),
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            drop_rate=0.,
            drop_path_rate=0.2,
            out_indices=(0, 1, 2, 3),
        )
        if imagenet_pretrained:
            _load_timm_swin_tiny_weights(self)

    def init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

        self.apply(_init)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


class SwinTinyUNet(nn.Module):
    def __init__(self, imagenet_pretrained=False):
        super().__init__()
        self.encoder = SwinTinyEncoder(imagenet_pretrained=imagenet_pretrained)
        self.dec1 = DecoderBlock(768, 384, 512)
        self.dec2 = DecoderBlock(512, 192, 256)
        self.dec3 = DecoderBlock(256, 96, 128)
        self.dec4 = DecoderBlock(128, 0, 64)
        self.dropout = nn.Dropout2d(0.3)
        self.head = nn.Conv2d(64, 2, kernel_size=1)
        self.register_buffer(
            'channel_mean',
            torch.zeros(1, len(_POLARIZATIONS), 1, 1),
        )
        self.register_buffer(
            'channel_std',
            torch.ones(1, len(_POLARIZATIONS), 1, 1),
        )

    def set_channel_normalization(self, means, stds):
        means = tuple(float(value) for value in means)
        stds = tuple(float(value) for value in stds)
        if len(means) != len(_POLARIZATIONS):
            raise ValueError(
                f'channel means must follow {_POLARIZATIONS}, got {means}'
            )
        if len(stds) != len(_POLARIZATIONS):
            raise ValueError(
                f'channel stds must follow {_POLARIZATIONS}, got {stds}'
            )
        if not all(math.isfinite(value) for value in means):
            raise ValueError(f'channel means must be finite, got {means}')
        if not all(math.isfinite(value) and value > 0 for value in stds):
            raise ValueError(
                f'channel stds must be finite and positive, got {stds}'
            )
        self.channel_mean.copy_(
            torch.tensor(means, dtype=self.channel_mean.dtype).view(
                1,
                len(_POLARIZATIONS),
                1,
                1,
            )
        )
        self.channel_std.copy_(
            torch.tensor(stds, dtype=self.channel_std.dtype).view(
                1,
                len(_POLARIZATIONS),
                1,
                1,
            )
        )
        return self

    def _prepare_input(self, x):
        if x.ndim != 4 or x.size(1) != len(_POLARIZATIONS):
            raise ValueError(
                'SwinTinyUNet expects VV+VH channels in shape [B, 2, H, W], '
                f'got {tuple(x.shape)}'
            )
        x = x.to(
            device=self.channel_mean.device,
            dtype=self.channel_mean.dtype,
        )
        return (x - self.channel_mean) / self.channel_std

    def forward(self, x):
        original_size = x.shape[-2:]
        x = self._prepare_input(x)
        f0, f1, f2, f3 = self.encoder(x)
        x = self.dec1(f3, f2)
        x = self.dec2(x, f1)
        x = self.dec3(x, f0)
        x = self.dec4(x)
        logits = self.head(self.dropout(x))
        return F.interpolate(
            logits,
            size=original_size,
            mode='bilinear',
            align_corners=False,
        )
