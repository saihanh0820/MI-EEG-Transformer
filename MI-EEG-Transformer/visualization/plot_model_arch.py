import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# 绘图
plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.axis('off')
ax.set_xlim(-1, 13)
ax.set_ylim(-1, 5)

# 输入层
input_rect = patches.Rectangle((0, 1.5), 2, 2, edgecolor="navy", facecolor="#E6F3FF", linewidth=2)
ax.add_patch(input_rect)
ax.text(1, 2.5, "Input Layer\n(500 × 22)\nTime × Channels", ha="center", va="center", fontsize=10, fontweight="bold", color="navy")

# 输入投影
ax.text(2.8, 2.5, "Projection\n(22 → 256)", ha="center", va="center", fontsize=9, style="italic")

# Transformer编码器
encoder_rect = patches.Rectangle((4, 0.5), 3, 4, edgecolor="darkred", facecolor="#FFE6E6", linewidth=2)
ax.add_patch(encoder_rect)
ax.text(5.5, 2.5, "Transformer Encoder\n(Electra-small)\n12 Layers | 256 Dim\nSelf-Attention", ha="center", va="center", fontsize=10, fontweight="bold", color="darkred")

# 分类头
classifier_rect = patches.Rectangle((8.5, 1.5), 2, 2, edgecolor="darkgreen", facecolor="#E6FFE6", linewidth=2)
ax.add_patch(classifier_rect)
ax.text(9.5, 2.5, "Classification Layer\n256 → 128 → 3\n(3 MI Tasks)", ha="center", va="center", fontsize=10, fontweight="bold", color="darkgreen")

# 箭头连接
arrow1 = FancyArrowPatch((2, 2.5), (3.5, 2.5), arrowstyle="->", linewidth=2, color="black", mutation_scale=15)
arrow2 = FancyArrowPatch((7, 2.5), (8.5, 2.5), arrowstyle="->", linewidth=2, color="black", mutation_scale=15)
ax.add_patch(arrow1)
ax.add_patch(arrow2)

# 总标题
ax.text(6.5, 4.5, "Architecture of the Proposed Transformer Model", fontsize=14, fontweight="bold", ha="center")

plt.tight_layout()
plt.savefig("../figures/Figure4_Model_Arch.png", dpi=300, bbox_inches='tight')
plt.close()