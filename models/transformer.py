import torch
import torch.nn as nn
from .afa import AdaptiveFeatureAttention

class EnhancedTransformerMAE(nn.Module):
    """
    AFA-Trans-ICL 的核心特征提取骨干。
    结合了 AFA 层和基于 [CLS] token 的 Transformer-MAE。
    """
    def __init__(self, input_dim, embed_dim=64, reduction_ratio=4):
        super().__init__()
        
        # 1. 引入独立拆分出的 AFA 模块
        self.afa = AdaptiveFeatureAttention(input_dim, reduction_ratio)
        self.feature_embed = nn.Linear(1, embed_dim)
        
        # 2. 引入 CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) 
        
        # 3. Transformer 编码器 (Norm-First)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, norm_first=True),
            num_layers=2
        )
        
        # 4. 轻量级解码器
        self.decoder = nn.Linear(embed_dim, 1)

    def forward(self, x, mask_ratio=0.3): 
        # AFA 特征校准
        attn_weights = self.afa(x)
        x_weighted = x * attn_weights
        
        # Token 嵌入
        tokens = self.feature_embed(x_weighted.unsqueeze(-1))
        
        # 掩码逻辑
        mask = None
        if mask_ratio > 0:
            mask = torch.rand(x.shape, device=x.device) < mask_ratio
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
            
        # 拼接 CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        # 编码
        encoded = self.encoder(tokens)
        
        # 提取隐空间表征 (仅提取 CLS token)
        latent = encoded[:, 0, :] 
        
        # 重建原始特征 (排除 CLS token)
        reconstruction = self.decoder(encoded[:, 1:, :]).squeeze(-1) 
        
        return latent, reconstruction, attn_weights, mask