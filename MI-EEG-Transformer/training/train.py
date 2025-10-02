import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from model.classifier import EEGTransformerClassifier
from training.config import Config

# 1. 自定义数据集类（加载Epoch数据）
class EEGDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32)  # (n_trials, 22, 500)
        self.y = torch.tensor(data['y'], dtype=torch.long)      # (n_trials,)
        # 调整维度为 (n_trials, 500, 22)（匹配Transformer输入：时间步×通道）
        self.X = self.X.permute(0, 2, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 动态阈值调整函数
def adjust_threshold(confidence_history, current_threshold, config):
    if len(confidence_history) < config.window_size:
        return current_threshold  # 不足5个样本，维持初始阈值
    # 计算最近5个样本的平均置信度
    avg_conf = np.mean(confidence_history[-config.window_size:])
    if avg_conf > 0.85:
        return min(current_threshold + 0.05, config.max_threshold)  # 提高阈值
    elif avg_conf < 0.65:
        return max(current_threshold - 0.05, config.min_threshold)  # 降低阈值
    else:
        return current_threshold  # 维持阈值

# 训练函数
def train_model(config):
    # 创建保存目录
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 加载数据（以Subject 1为例，可循环加载6个受试者数据）
    data_path = os.path.join(config.data_dir, "A01T_epochs.npz")
    dataset = EEGDataset(data_path)
    # 分割训练集/验证集
    train_dataset, val_dataset = train_test_split(dataset, test_size=config.val_split, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGTransformerClassifier(
        input_dim=config.n_channels,
        hidden_dim=256,
        num_classes=config.num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，与论文一致
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=config.weight_decay
    )
    # 学习率调度
    def adjust_learning_rate(optimizer, epoch, config):
        if epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr_decay

    # 初始化训练状态
    best_val_acc = 0.0
    current_threshold = config.initial_threshold
    confidence_history = []  # 存储置信度历史，用于调整阈值

    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        # 调整学习率（第5轮）
        adjust_learning_rate(optimizer, epoch, config)

        # 训练批次
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            X, y = batch
            X, y = X.to(device), y.to(device)

            # 前向传播
            logits, probs, confidence = model(X)
            loss = criterion(logits, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失和预测结果
            train_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            # 记录置信度（用于动态阈值）
            confidence_history.extend(confidence.detach().cpu().numpy())

        # 调整动态阈值
        current_threshold = adjust_threshold(confidence_history, current_threshold, config)

        # 计算训练集指标
        train_loss_avg = train_loss / len(train_dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        #  验证集评估
        model.eval()
        val_preds = []
        val_labels = []
        val_confidence = []
        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                X, y = X.to(device), y.to(device)
                logits, probs, confidence = model(X)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
                val_confidence.extend(confidence.cpu().numpy())

        # 计算验证集指标
        val_acc = accuracy_score(val_labels, val_preds)
        val_conf_avg = np.mean(val_confidence)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, "best_model.pth"))
            print(f"Best model saved! Val Acc: {best_val_acc:.4f}")

        # 打印日志
        log_str = f"Epoch {epoch+1} | LR: {optimizer.param_groups[0]['lr']:.6f} | " \
                  f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | " \
                  f"Val Acc: {val_acc:.4f} | Val Conf Avg: {val_conf_avg:.4f} | " \
                  f"Current Threshold: {current_threshold:.2f}"
        print(log_str)
        # 写入日志文件
        with open(os.path.join(config.log_dir, "train_log.txt"), "a") as f:
            f.write(log_str + "\n")

    return model

# 主函数（执行训练）
if __name__ == "__main__":
    config = Config()
    train_model(config)