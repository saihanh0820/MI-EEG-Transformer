import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# 绘图
plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
ax.axis('off')
ax.set_xlim(-1, 16)
ax.set_ylim(-2, 6)

# 步骤1：置信度计算
step1_rect = patches.Rectangle((0, 2), 3, 2, edgecolor="navy", facecolor="#E6F3FF", linewidth=2)
ax.add_patch(step1_rect)
ax.text(1.5, 3, "Step 1: Confidence Calculation\nMax(3-Class Probabilities)\nExample: P(Left)=0.85 → Conf=0.85", ha="center", va="center", fontsize=10, fontweight="bold", color="navy")

# 步骤2：动态阈值
step2_rect = patches.Rectangle((5, 2), 3.5, 2, edgecolor="darkred", facecolor="#FFE6E6", linewidth=2)
ax.add_patch(step2_rect)
ax.text(6.75, 3, "Step 2: Dynamic Threshold\nInitial=0.7\n- Avg(5 samples) >0.85 → 0.75\n- Avg(5 samples) <0.65 → 0.65", ha="center", va="center", fontsize=10, fontweight="bold", color="darkred")

# 步骤3：反馈调整
step3_rect = patches.Rectangle((10, 2), 3.5, 2, edgecolor="darkgreen", facecolor="#E6FFE6", linewidth=2)
ax.add_patch(step3_rect)
ax.text(11.75, 3, "Step 3: Feedback Adjustment\n- Conf ≥ Threshold → Output\n- Conf < Threshold → Resample + Vote", ha="center", va="center", fontsize=10, fontweight="bold", color="darkgreen")

# 箭头连接（闭环）
arrow1 = FancyArrowPatch((3, 3), (5, 3), arrowstyle="->", linewidth=2, color="black", mutation_scale=15)
arrow2 = FancyArrowPatch((8.5, 3), (10, 3), arrowstyle="->", linewidth=2, color="black", mutation_scale=15)
arrow3 = FancyArrowPatch((11.75, 2), (6.75, 1), arrowstyle="->", linewidth=2, color="gray", linestyle="--", mutation_scale=12)
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.add_patch(arrow3)

# 闭环标注
ax.text(8.5, 0.5, "Closed-Loop: Update Threshold with New Samples", ha="center", fontsize=9, style="italic", color="gray")

# 总标题
ax.text(7.5, 5.2, "Cybernetics-Based Closed-Loop Feedback Mechanism", fontsize=14, fontweight="bold", ha="center")

plt.tight_layout()
plt.savefig("../figures/Figure5_Feedback.png", dpi=300, bbox_inches='tight')
plt.close()