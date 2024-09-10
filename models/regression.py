import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_pca import PCA

import torch

def pca_reduction(data, n_components):
    # 标准化数据
    data_mean = torch.mean(data, dim=0, keepdim=True)
    data_std = torch.std(data, dim=0, keepdim=True)
    data_normalized = (data - data_mean) / data_std

    # 计算协方差矩阵
    cov_matrix = torch.cov(data_normalized.view(-1, data.size(-1)).T)

    # 计算特征值和特征向量
    cov_matrix = cov_matrix.to(dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    eigenvectors = eigenvectors.to(dtype=torch.float16)

    # 选择前 n_components 个特征向量
    top_eigenvectors = eigenvectors[:, -n_components:]

    # 投影数据
    reduced_data = torch.matmul(data_normalized.view(-1, data.size(-1)), top_eigenvectors)

    # 调整形状为 (32, 1, 100)
    # reduced_data = reduced_data.view(data.size(0), data.size(1), n_components)
    reduced_data = reduced_data.view(data.size(0), n_components)

    return reduced_data

class MHA(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(out_dim, num_heads=8, batch_first=True)
        self.key = nn.Linear(in_dim, out_dim)
        self.query = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.layernorm = nn.LayerNorm(in_dim)

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
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_0 = self.model(x)
        x = x + x_0
        return x

class Regression(nn.Module):
    def __init__(self, in_dim, regression_ratio=0.5):
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
        self.mid_dim = mid_dim
        self.l_pre = nn.Linear(in_dim, mid_dim)
        # self.mha1 = MHA(in_dim, in_dim)
        # self.mlp1 = MLP(in_dim)
        self.mha1 = MHA(mid_dim, mid_dim)
        self.mlp1 = MLP(mid_dim)
        # self.l0 = nn.Linear(in_dim, mid_dim)
        # self.mha2 = MHA(mid_dim, mid_dim)
        # self.mlp2 = MLP(mid_dim)
        # self.l1 = nn.Linear(mid_dim, end_dim)
        # self.l4 = nn.Linear(in_dim, mid_dim)
        # self.l2 = nn.Linear(end_dim, 8)
        # self.l3 = nn.Linear(8, 1)
        # self.l5 = nn.Linear(mid_dim, 1)
        self.l4 = nn.Linear(mid_dim, 8)
        
        
    def forward(self, x):
        # h = self.regress(x)
        # x_reduced = pca_reduction(x, self.mid_dim)

        x_reduced = self.l_pre(x)

        # x_1 = self.mha1(x)
        x_1 = self.mha1(x_reduced)
        x_2 = self.mlp1(x_1)
        # x_2 = self.l0(x_2)
        # x_3 = self.mha2(x_2)
        # x_4 = self.mlp2(x_3)
        # h = self.l1(x_4)
        h = self.l4(x_2)
        # h = self.l2(h)
        # h = self.l3(h)
        # h = self.l5(h)
        h = torch.matmul(h.view(x.size(0),1,-1), h.view(x.size(0),1,-1).transpose(1, 2))
        h = h.squeeze(dim=1)
        return h
        
    
def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')

if __name__ == '__main__':
    # x = torch.randn(32, 2560)
    # m = Regression(in_dim=2560
    #                )
    x = torch.randn(32, 1536)
    m = Regression(in_dim=1536
                   )
    r = m(x)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')