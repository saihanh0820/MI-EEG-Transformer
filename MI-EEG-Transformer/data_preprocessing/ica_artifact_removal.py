import mne
import numpy as np

def ica_remove_eog(raw_eeg_path, save_path):
    """
    输入：raw_eeg_path（原始EEG文件路径，如GDF/EDF格式）、save_path（去伪影后保存路径）
    输出：去伪影后的EEG数据
    """
    # 加载原始EEG数据
    raw = mne.io.read_raw_gdf(raw_eeg_path, preload=True)
    # 仅保留22个头皮电极（排除EOG/肌电通道）
    raw.pick_types(eeg=True)
    # 重命名电极（匹配10-20系统，方便后续定位）
    raw.rename_channels({'EEG-Fz': 'Fz', 'EEG-0': 'C3', 'EEG-1': 'Cz', 'EEG-2': 'C4',
                         'EEG-3': 'F3', 'EEG-4': 'F4', 'EEG-5': 'Pz', 'EEG-6': 'F7',
                         'EEG-7': 'F8', 'EEG-8': 'T7', 'EEG-9': 'T8', 'EEG-10': 'P3',
                         'EEG-11': 'P4', 'EEG-12': 'O1', 'EEG-13': 'O2', 'EEG-14': 'AF3',
                         'EEG-15': 'AF4', 'EEG-16': 'FC1', 'EEG-17': 'FC2', 'EEG-18': 'FC5',
                         'EEG-19': 'FC6', 'EEG-20': 'CP1', 'EEG-21': 'CP2'})

    # 运行ICA（15个成分，与论文一致）
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw)

    # 识别EOG伪影（用前额头电极Fp1/Fp2，论文2.1.2节）
    eog_indices, _ = ica.find_bads_eog(raw, ch_name=['AF3', 'AF4'])  # AF3/AF4近似Fp1/Fp2功能
    ica.exclude = eog_indices  # 标记需去除的伪影成分

    # 应用ICA去伪影
    raw_ica = ica.apply(raw)

    # 保存去伪影后的数据
    raw_ica.save(save_path, overwrite=True)
    print(f"ICA去伪影完成，保存路径：{save_path}")
    return raw_ica


if __name__ == "__main__":
    # 替换为你的原始EEG文件路径（如BCI Competition IV的A01T.gdf）
    raw_path = "../data/A01T.gdf"
    # 去伪影后保存路径
    save_path = "../data/A01T_ica.fif"
    # 执行去伪影
    ica_remove_eog(raw_path, save_path)