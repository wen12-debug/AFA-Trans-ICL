import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path, sample_size=None):
    """
    加载 Edge-IIoTset，清洗无用特征并进行编码转换
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据集文件: {path}，请检查路径是否正确！")
        
    print(f"[{pd.Timestamp.now()}] 开始读取数据...")
    df = pd.read_csv(path, low_memory=False)
    if sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # 剔除无业务意义或全为空的特征
    drop_cols = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 
                 'arp.dst.proto_ipv4', 'http.file_data', 'http.request.full_uri', 
                 'tcp.options', 'tcp.payload', 'mqtt.msg']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    # 异常值清理
    df.replace(' ', np.nan, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(f"[{pd.Timestamp.now()}] 开始类别特征编码...")
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['Attack_label', 'Attack_type']:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
    y = df['Attack_label'].values.astype(int)
    X_raw = df.drop(['Attack_label', 'Attack_type'], axis=1, errors='ignore').values
    
    return X_raw, y