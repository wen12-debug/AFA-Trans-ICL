import torch.nn as nn

class AdaptiveFeatureAttention(nn.Module):
    """
    论文中的 AFA (Adaptive Feature Attention) 模块。
    作为信息瓶颈，动态校准 IoT 流量特征重要性。
    """
    def __init__(self, input_dim, reduction_ratio=4):
        super().__init__()
        reduced_dim = max(4, input_dim // reduction_ratio)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attention(x)