import mne
import matplotlib.pyplot as plt
import numpy as np

# 加载原始数据和滤波后数据
raw_path = "../data/A01T.gdf"
filtered_path = "../data/A01T_filtered.fif"
raw = mne.io.read_raw_gdf(raw_path, preload=True)
filtered = mne.io.read_raw_fif(filtered_path, preload=True)

# 提取Fp1电极的信号（前额头电极，易受线噪影响）
ch_name = "AF3"  # 近似Fp1
raw_data = raw.get_data(picks=[ch_name])[0]
filtered_data = filtered.get_data(picks=[ch_name])[0]
times = raw.times[:1000]  # 取前4秒数据（避免图过大）
raw_data = raw_data[:1000]
filtered_data = filtered_data[:1000]

# 绘图
plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(12, 6))
plt.plot(times, raw_data, color="darkgray", label="Raw Signal (with 50Hz noise)", alpha=0.8)
plt.plot(times, filtered_data, color="darkblue", label="Filtered Signal (8-30Hz Chebyshev Type I)")
# 标注50Hz噪声段
plt.axvspan(1.0, 1.5, color="red", alpha=0.1, label="50Hz Noise Segment")
plt.xlabel("Time (s)", fontsize=11)
plt.ylabel("Amplitude (μV)", fontsize=11)
plt.title("Comparison of Raw and Filtered EEG Signals (AF3 Electrode)", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig("../figures/Figure2_Raw_Filtered.png", dpi=300, bbox_inches='tight')
plt.close()