import torch.nn as nn

def calculate_joint_loss(recon, x_b, mask, attn_weights, lambda_val=0.001):
    """计算 MAE 重建损失与 AFA 的 L1 稀疏损失的联合损失"""
    if mask is not None and mask.sum() > 0:
        mse_loss = nn.MSELoss(reduction='sum')(recon[mask], x_b[mask]) / mask.sum()
    else:
        mse_loss = nn.MSELoss()(recon, x_b)
        
    l1_loss = attn_weights.abs().mean()
    
    total_loss = mse_loss + lambda_val * l1_loss
    return total_loss