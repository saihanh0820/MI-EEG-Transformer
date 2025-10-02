import numpy as np
import matplotlib.pyplot as plt

# 加载Epoch数据
data_path = "../data/A01T_epochs.npz"
data = np.load(data_path)
X = data['X']  # (216, 22, 500)
y = data['y']  # 0=左手，1=右手，2=双脚
times = np.linspace(1.0, 3.0, 500)  # 1-3秒时间轴

# 选择3个典型通道（F3/C3/F4，运动想象相关脑区）
ch_indices = [3, 1, 4]  # F3/C3/F4在22通道中的索引（需根据实际电极顺序调整）
ch_names = ["F3", "C3", "F4"]

# 分别取3类任务的1个Epoch（取第1个样本）
left_epoch = X[np.where(y==0)[0][0], ch_indices, :]  # 左手
right_epoch = X[np.where(y==1)[0][0], ch_indices, :]  # 右手
feet_epoch = X[np.where(y==2)[0][0], ch_indices, :]  # 双脚

# 绘图
plt.rcParams['font.family'] = 'Arial'
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
task_data = [left_epoch, right_epoch, feet_epoch]
task_labels = ["Left Hand MI", "Right Hand MI", "Feet MI"]
colors = ["darkred", "darkblue", "darkgreen"]

for ax, data, label, color in zip(axes, task_data, task_labels, colors):
    for i, (ch_data, ch_name) in enumerate(zip(data, ch_names)):
        ax.plot(times, ch_data, color=color, alpha=0.6 + 0.2*i, label=f"Channel {ch_name}")
    ax.set_title(label, fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Amplitude (μV)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1.0, 3.0)

# 总标题
fig.suptitle("Example Epoch Waveforms of Motor Imagery Tasks", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../figures/Figure3_Epoch_Waveforms.png", dpi=300, bbox_inches='tight')
plt.close()