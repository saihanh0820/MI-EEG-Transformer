import mne
import numpy as np
from scipy.signal import cheby1, filtfilt

def chebyshev_bandpass_filter(raw_ica_path, save_path, l_freq=8, h_freq=30, order=4, ri=0.5):
    """
    输入：raw_ica_path（ICA去伪影后的EEG路径）、save_path（滤波后保存路径）
    输出：滤波后的EEG数据
    """
    # 加载ICA去伪影后的数据
    raw = mne.io.read_raw_fif(raw_ica_path, preload=True)
    # 提取数据矩阵（shape: (n_channels, n_timepoints)）
    data = raw.get_data()
    fs = raw.info['sfreq']  # 采样率（250Hz，与论文一致）

    # 设计Chebyshev I型带通滤波器
    nyq = 0.5 * fs  # 奈奎斯特频率
    low_norm = l_freq / nyq
    high_norm = h_freq / nyq
    b, a = cheby1(order, ri, [low_norm, high_norm], btype='band')

    # 零相位滤波（避免信号相位偏移）
    filtered_data = filtfilt(b, a, data, axis=1)  # 沿时间轴滤波

    # 重构MNE Raw对象（保留电极信息）
    raw_filtered = mne.io.RawArray(filtered_data, raw.info)

    # 保存滤波后的数据
    raw_filtered.save(save_path, overwrite=True)
    print(f"Chebyshev滤波完成，保存路径：{save_path}")
    return raw_filtered

# 示例用法
if __name__ == "__main__":
    # ICA去伪影后的文件路径
    ica_path = "../data/A01T_ica.fif"
    # 滤波后保存路径
    save_path = "../data/A01T_filtered.fif"
    # 执行滤波（参数与论文一致：8-30Hz，order=4，ri=0.5）
    chebyshev_bandpass_filter(ica_path, save_path)