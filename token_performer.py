import math
import torch
import torch.nn as nn


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1, gamma=False, init_values=1e-5):
        super().__init__()
        self.head_dim = in_dim // head_cnt
        self.emb = in_dim
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8 
        self.drop_path = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.head_dim * kernel_ratio)
        self.w = torch.randn(head_cnt, self.m, self.head_dim)
        for i in range(self.head_cnt):
            self.w[i] = nn.Parameter(nn.init.orthogonal_(self.w[i]) * math.sqrt(self.m))
        self.w.requires_grad_(False)

        if gamma:
            self.gamma1 = nn.Parameter(init_values * torch.ones((self.emb)))
        else:
            self.gamma1 = 1

    def prm_exp(self, x):

        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, 1, self.m) / 2
        wtx = torch.einsum('bhti,hmi->bhtm', x.float(), self.w.to(x.device))

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        B, N, C = x.shape
        kqv = self.kqv(x).reshape(B, N, 3, self.head_cnt, self.head_dim).permute(2, 0, 3, 1, 4)
        k, q, v = kqv[0], kqv[1], kqv[2] 

        kp, qp = self.prm_exp(k), self.prm_exp(q)  
        D = torch.einsum('bhti,bhi->bht', qp, kp.sum(dim=2)).unsqueeze(dim=-1)
        kptv = torch.einsum('bhin,bhim->bhnm', v.float(), kp)
        y = torch.einsum('bhti,bhni->bhtn', qp, kptv) / (D.repeat(1, 1, 1, self.head_dim) + self.epsilon)

        y = y.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        v = v.permute(0, 2, 1, 3).reshape(B, N, self.emb)

        y = v + self.dp(self.gamma1 * self.proj(y)) 

        return y

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x