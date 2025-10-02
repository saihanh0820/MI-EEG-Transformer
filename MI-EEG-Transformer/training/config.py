class Config:
    # 数据参数
    n_channels = 22  # 22个电极，与论文一致
    n_timepoints = 500  # 500个时间点（1-3秒，250Hz）
    num_classes = 3  # 左手/右手/双脚，与论文一致

    # 训练参数
    epochs = 10  # 训练10轮
    batch_size = 16  # 批次大小16
    initial_lr = 1e-4  # 初始学习率1e-4
    lr_decay = 5e-5  # 第5轮调整为5e-5
    weight_decay = 1e-5  # 权重衰减，防止过拟合
    val_split = 0.15  # 验证集比例15%（与论文5折交叉验证一致）

    # 反馈机制参数
    initial_threshold = 0.7  # 初始阈值0.7
    window_size = 5  # 5个样本的平均置信度调整阈值
    max_threshold = 0.75  # 最高阈值0.75
    min_threshold = 0.65  # 最低阈值0.65

    # 路径参数
    data_dir = "~/data/"  # 数据存放目录
    model_save_dir = "~/saved_models/"  # 模型保存目录
    log_dir = "~/logs/"  # 日志保存目录