import os
import argparse
import numpy as np
import torch
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils.seed import set_seed
from data.dataset import preprocess_data
from models.transformer import EnhancedTransformerMAE

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AFA-Trans-ICL")
    parser.add_argument('--data_path', type=str, required=True, help="Path to Edge-IIoTset CSV")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory of saved models")
    parser.add_argument('--sample_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1024)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载数据构建测试集
    X_raw, y = preprocess_data(args.data_path, args.sample_size)
    X_normal = X_raw[y == 0]
    X_attack = X_raw[y == 1]
    
    _, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=42)
    X_test = np.vstack((X_test_normal, X_attack))
    y_test = np.concatenate((np.zeros(len(X_test_normal)), np.ones(len(X_attack))))

    # 2. 加载预训练模型组件
    scaler_path = os.path.join(args.save_dir, 'robust_scaler.pkl')
    mae_path = os.path.join(args.save_dir, 'afa_trans_mae.pth')
    clf_path = os.path.join(args.save_dir, 'icl_clf.pkl')

    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.clip(X_test_scaled, -10.0, 10.0).astype(np.float32)

    input_dim = X_test_scaled.shape[1]
    model = EnhancedTransformerMAE(input_dim).to(device)
    model.load_state_dict(torch.load(mae_path, map_location=device))
    model.eval()
    
    clf = joblib.load(clf_path)

    # 3. 推断
    print("\n[Phase 3] 开始测试集推断...")
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled)), batch_size=args.batch_size, shuffle=False)
    
    Z_test_list = []
    with torch.no_grad():
        for batch in test_loader:
            x_b = batch[0].to(device)
            z_b, _, _, _ = model(x_b, mask_ratio=0.0) 
            Z_test_list.append(z_b.cpu().numpy())
            
    Z_test = np.concatenate(Z_test_list, axis=0)
    scores = clf.decision_function(Z_test)

    # 4. 评估
    auc_roc = roc_auc_score(y_test, scores)
    ap_score = average_precision_score(y_test, scores)
    
    print("\n" + "="*40)
    print(f"最终评估指标 (OSAD 规范)")
    print(f"AUROC : {auc_roc:.4f}")
    print(f"AP    : {ap_score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()