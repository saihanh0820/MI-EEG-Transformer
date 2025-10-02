import mne
import numpy as np

def segment_epochs(raw_filtered_path, save_npy_path, tmin=1.0, tmax=3.0):
    """
    输入：raw_filtered_path（滤波后EEG路径）、save_npy_path（Epoch保存路径，npy格式）
    输出：Epoch数据（shape: (n_trials, 22, 500)）和标签
    """
    # 加载滤波后的数据
    raw = mne.io.read_raw_fif(raw_filtered_path, preload=True)
    # 提取事件（BCI Competition IV的事件编码：7=左手，8=右手，9=双脚）
    events, event_id = mne.events_from_annotations(raw)
    # 筛选3类运动想象任务的事件（排除舌头任务）
    target_event_id = {'left_hand': 7, 'right_hand': 8, 'feet': 9}

    # 分割Epoch（时间窗口：1-3秒，与论文一致）
    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_event_id,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        reject_by_annotation=False  # 不自动剔除坏段（后续手动处理）
    )

    # 提取Epoch数据和标签
    X = epochs.get_data()  # shape: (n_trials, n_channels, n_timepoints) → (216, 22, 500)
    y = epochs.events[:, 2]  # 标签：7=左手，8=右手，9=双脚
    # 标签映射为0/1/2（方便模型训练）
    y_map = {7: 0, 8: 1, 9: 2}
    y = np.array([y_map[label] for label in y])

    # 4. Z-score归一化（按时间步归一化，与论文2.1.2节一致）
    X = (X - np.mean(X, axis=2, keepdims=True)) / np.std(X, axis=2, keepdims=True)

    # 5. 保存为npy格式（方便后续加载）
    np.savez(save_npy_path, X=X, y=y)
    print(f"Epoch分割完成，数据形状：{X.shape}，标签形状：{y.shape}")
    print(f"保存路径：{save_npy_path}")
    return X, y

# 示例用法
if __name__ == "__main__":
    # 滤波后的文件路径
    filtered_path = "../data/A01T_filtered.fif"
    # Epoch保存路径（npz格式，含数据和标签）
    save_path = "../data/A01T_epochs.npz"
    # 执行分割
    segment_epochs(filtered_path, save_path)