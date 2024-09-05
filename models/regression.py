import torch.nn as nn
import torch
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(out_dim, num_heads=8, batch_first=True)
        self.key = nn.Linear(in_dim, out_dim)
        self.query = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x_n = self.layernorm(x)
        q = self.query(x_n)
        k = self.key(x_n)
        v = self.value(x_n)
        x_o, _ = self.mha(q, k, v)    # 注意力机制的输出是一个元组，第一个元素是输出，第二个是注意力权重
        x = x + x_o

        return x


class MLP(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super(MLP, self).__init__()
        mid_dim = int(in_dim * mlp_ratio)

        self.model = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_0 = self.model(x)
        x = x + x_0
        return x

class Regression(nn.Module):
    def __init__(self, in_dim, regression_ratio=0.8):
        super().__init__()
        mid_dim = int(in_dim * regression_ratio)
        end_dim = int(mid_dim * regression_ratio)
        self.regress = nn.Sequential(
            MHA(in_dim, in_dim),
            MLP(in_dim, mlp_ratio=regression_ratio),
            MHA(mid_dim, mid_dim),
            MLP(in_dim, mlp_ratio=regression_ratio),
            nn.Linear(end_dim, 8),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        h = self.regress(x)
        return h
        
    
