import mne
import matplotlib.pyplot as plt

# 加载ICA去伪影后的数据（含ICA信息）
raw_ica_path = "../data/A01T_ica.fif"
raw = mne.io.read_raw_fif(raw_ica_path, preload=True)
# 提取ICA对象（需确保之前的ICA结果已保存，或重新运行ICA）
ica = mne.preprocessing.read_ica("../data/A01T_ica.fif")  # 若未单独保存，需重新训练ICA

# 绘制前6个ICA成分（标注伪影成分）
plt.rcParams['font.family'] = 'Arial'  # 学术期刊字体
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

# 绘制每个成分
for i, ax in enumerate(axes):
    ica.plot_components(picks=[i], axes=ax, show=False)
    ax.set_title(f"Component {i+1}" + (" (EOG Artifact)" if i in ica.exclude else " (Brain Activity)"), 
                 fontsize=10, color="red" if i in ica.exclude else "black")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude (μV)", fontsize=9)

# 总标题和保存
fig.suptitle("ICA Components Identified as Artifacts", fontsize=14, fontweight="bold")
fig.text(0.5, 0.02, "Artifact components (red) show abnormal amplitude fluctuations", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("../figures/Figure1_ICA_Artifacts.png", dpi=300, bbox_inches='tight')
plt.close()