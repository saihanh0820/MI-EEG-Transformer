import torch
import torch.nn as nn
from model.transformer import EEGTransformerEncoder

class EEGTransformerClassifier(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=256, num_classes=3):
        super().__init__()
        # 1. 加载Transformer编码器
        self.encoder = EEGTransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        # 2. 分类头（与论文一致：256→128→3，含ReLU和Dropout）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 防止过拟合，与论文一致
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (BS, 500, 22)
        # 编码器输出全局特征
        global_feat = self.encoder(x)  # (BS, 256)
        # 分类输出logits
        logits = self.classifier(global_feat)  # (BS, 3)
        # 输出logits和置信度（置信度用于反馈机制）
        probs = torch.softmax(logits, dim=1)  # (BS, 3)
        confidence = torch.max(probs, dim=1).values  # (BS,) 每个样本的最大置信度
        return logits, probs, confidence