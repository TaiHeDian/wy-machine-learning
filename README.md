# EMG信号分类深度学习项目

这是一个使用PyTorch实现的EMG（肌电信号）一维时序信号分类项目。该项目面向零基础用户，代码包含详细的中文注释。

## 项目概述

本项目实现了从数据读取、预处理、降噪、片段提取，到模型训练和结果可视化的完整流程。主要用于论文展示EMG信号分类效果。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- 其他依赖见 requirements.txt

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据格式

数据文件应放置在 `data/` 文件夹中，每个类别对应一个或多个 .txt 文件。

文件格式：

- 前3行：表头信息（跳过）
- 从第4行开始：序号、时间(MM:SS.ms)、电压(mV)、事件
- 列分隔符：空格或制表符
- 采样率：1000Hz

## 运行项目

直接运行主文件：

```bash
python main.py
```

程序将自动：

1. 读取和预处理所有数据文件
2. 构建训练和测试数据集
3. 训练1D-CNN模型
4. 生成4张可视化图像

## 输出文件

- `emg_cnn_model.pth`：训练好的模型权重
- `preprocessing_comparison.png`：信号预处理对比图
- `training_curves.png`：训练过程曲线
- `confusion_matrix.png`：混淆矩阵热力图
- `tsne_visualization.png`：t-SNE特征可视化

## 代码结构说明

- `read_emg_file()`：安全读取EMG数据文件
- `denoise_signal()`：信号降噪（50Hz陷波 + 20-400Hz带通）
- `extract_active_segments()`：基于RMS的活动段提取
- `EMG_Dataset`：PyTorch数据集类
- `EMG_CNN`：1D-CNN模型类
- `train_model()`：模型训练函数
- 各种绘图函数：生成可视化图像

## 注意事项

1. 确保数据文件格式正确
2. 如果没有GPU，将自动使用CPU训练
3. 训练过程可能需要几分钟到几十分钟，取决于数据量
4. 生成的图像保存在项目根目录下

## 自定义参数

可以在 `main()` 函数中修改以下参数：

- `segment_length`：片段长度（默认1000）
- `num_epochs`：训练轮数（默认100）
- `batch_size`：批次大小（默认32）
- 滤波器参数等
