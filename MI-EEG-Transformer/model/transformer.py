import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraConfig

class EEGTransformerEncoder(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=256, num_layers=12, num_heads=8):
        super().__init__()
        # 1. 配置Electra-small编码器（与论文一致：12层，隐藏维256）
        self.config = ElectraConfig(
            vocab_size=1,  # 非文本数据，无需词汇表
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=1024,  # Electra-small默认中间层维度
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        # 2. 加载Electra编码器（无预训练权重，从头训练，与论文“简化Transformer”一致）
        self.encoder = ElectraModel(self.config)
        # 3. 输入投影层（将22通道映射到256维，匹配编码器输入）
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x: 输入形状 (batch_size, time_steps, input_dim) → (BS, 500, 22)
        # 投影到256维
        x_proj = self.input_proj(x)  # (BS, 500, 256)
        # Transformer编码（取最后一层输出）
        encoder_outputs = self.encoder(inputs_embeds=x_proj)
        last_hidden_state = encoder_outputs.last_hidden_state  # (BS, 500, 256)
        # 取全局特征（简化为最后一个时间步的输出）
        global_feat = last_hidden_state[:, -1, :]  # (BS, 256)
        return global_feat