import os
import argparse
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
from deepod.models.icl import ICL
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# 导入我们刚刚拆分的模块
from utils.seed import set_seed
from utils.loss import calculate_joint_loss
from data.dataset import preprocess_data
from models.transformer import EnhancedTransformerMAE

def parse_args():
    parser = argparse.ArgumentParser(description="Train AFA-Trans-ICL")
    parser.add_argument('--data_path', type=str, required=True, help="Path to Edge-IIoTset CSV")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory to save models")
    parser.add_argument('--sample_size', type=int, default=100000, help="Number of samples to use")
    parser.add_argument('--epochs', type=int, default=30, help="MAE pre-training epochs")
    parser.add_argument('--icl_epochs', type=int, default=50, help="ICL optimization epochs")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_val', type=float, default=0.001, help="Sparsity penalty weight")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"--- 训练环境初始化 ---")
    print(f"Device: {device} | Mask Ratio: {args.mask_ratio} | Lambda: {args.lambda_val}")

    # 1. 数据准备
    X_raw, y = preprocess_data(args.data_path, args.sample_size)
    X_normal = X_raw[y == 0]
    X_train_normal, _ = train_test_split(X_normal, test_size=0.2, random_state=args.seed)

    # 2. 鲁棒缩放
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_train_scaled = np.clip(X_train_scaled, -10.0, 10.0).astype(np.float32)
    joblib.dump(scaler, os.path.join(args.save_dir, 'robust_scaler.pkl'))

    input_dim = X_train_scaled.shape[1]

    # 3. Phase 1: MAE 训练
    print("\n========== [Phase 1] 开始 MAE 自监督训练 ==========")
    model = EnhancedTransformerMAE(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled)), batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in train_loader:
            x_b = batch[0].to(device)
            latent, recon, attn_weights, mask = model(x_b, mask_ratio=args.mask_ratio) 
            
            loss = calculate_joint_loss(recon, x_b, mask, attn_weights, args.lambda_val)
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {epoch_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'afa_trans_mae.pth'))

    # 4. Phase 2: ICL 边界学习
    print("\n========== [Phase 2] 开始 ICL 边界优化 ==========")
    model.eval()
    Z_train_list = []
    with torch.no_grad():
        for batch in train_loader:
            x_b = batch[0].to(device)
            z_b, _, _, _ = model(x_b, mask_ratio=0.0) 
            Z_train_list.append(z_b.cpu().numpy())
            
    Z_train_normal = np.concatenate(Z_train_list, axis=0)
    clf = ICL(n_ensemble=5, epochs=args.icl_epochs, device=device, random_state=args.seed) 
    clf.fit(Z_train_normal)
    joblib.dump(clf, os.path.join(args.save_dir, 'icl_clf.pkl'))
    print("模型与预处理文件已全部保存至 checkpionts 目录。")

if __name__ == "__main__":
    main()