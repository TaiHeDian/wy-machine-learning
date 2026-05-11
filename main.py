import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import random

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 数据读取与预处理函数
def read_emg_file(file_path):
    """
    安全读取EMG数据文件，提取电压信号。

    参数:
    file_path (str): 文件路径

    返回:
    np.ndarray: 一维电压信号数组 (mV)
    """
    encodings = ['utf-8', 'gbk', 'latin1']  # 尝试多种编码
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"无法读取文件 {file_path}，尝试了编码: {encodings}")
        return np.array([])

    # 跳过前3行表头
    data_lines = lines[3:]

    voltages = []
    for line in data_lines:
        if line.strip():  # 跳过空行
            parts = line.split()  # 用空格或制表符分割
            if len(parts) >= 3:  # 确保有足够的列
                try:
                    voltage = float(parts[2])  # 第3列是电压 (mV)
                    voltages.append(voltage)
                except ValueError:
                    continue  # 跳过无法转换的行

    return np.array(voltages)

# 信号降噪函数
def denoise_signal(signal_data, fs=1000):
    """
    对EMG信号进行降噪处理：50Hz陷波滤波 + 20-400Hz带通滤波。

    参数:
    signal_data (np.ndarray): 原始信号
    fs (int): 采样率，默认1000Hz

    返回:
    np.ndarray: 降噪后的信号
    """
    if len(signal_data) < 100:  # 如果信号太短，返回原信号
        return signal_data

    # 50Hz陷波滤波器去除工频干扰
    notch_freq = 50.0
    quality_factor = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
    signal_notched = signal.filtfilt(b_notch, a_notch, signal_data)

    # 20-400Hz四阶巴特沃斯带通滤波器
    lowcut = 20.0
    highcut = 400.0
    order = 4
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b_band, a_band = signal.butter(order, [low, high], btype='band')
    signal_filtered = signal.filtfilt(b_band, a_band, signal_notched)

    return signal_filtered

# 有效片段提取函数
def extract_active_segments(signal_data, fs=1000, window_size=100, threshold_factor=2.0, segment_length=1000):
    """
    基于滑动窗口RMS的端点检测，提取活动段。

    参数:
    signal_data (np.ndarray): 降噪后的信号
    fs (int): 采样率
    window_size (int): 滑动窗口大小 (样本点)
    threshold_factor (float): 阈值因子
    segment_length (int): 截取片段长度

    返回:
    list: 活动片段列表，每个片段为np.ndarray
    """
    # 计算RMS能量
    rms = np.sqrt(np.convolve(signal_data**2, np.ones(window_size)/window_size, mode='valid'))

    # 计算阈值
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    threshold = mean_rms + threshold_factor * std_rms

    # 找出超过阈值的窗口
    active_indices = np.where(rms > threshold)[0]

    segments = []
    if len(active_indices) > 0:
        # 合并相邻的活动窗口
        diff = np.diff(active_indices)
        split_indices = np.where(diff > window_size)[0] + 1
        active_groups = np.split(active_indices, split_indices)

        for group in active_groups:
            if len(group) > 0:
                start = group[0]
                end = group[-1] + window_size

                # 找到峰值位置
                peak_idx = start + np.argmax(rms[start:end])

                # 以峰值为中心截取固定长度
                half_length = segment_length // 2
                seg_start = max(0, peak_idx - half_length)
                seg_end = min(len(signal_data), peak_idx + half_length)

                if seg_end - seg_start == segment_length:
                    segment = signal_data[seg_start:seg_end]
                    segments.append(segment)

    return segments

# 数据集类
class EMG_Dataset(Dataset):
    """
    EMG数据集类
    """
    def __init__(self, data_list, labels):
        self.data = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        # 转换为Tensor，形状 (1, length)
        signal_tensor = torch.tensor(signal.copy(), dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return signal_tensor, label_tensor

# 构建数据集
def build_dataset(data_dir='data', fs=1000, segment_length=1000):
    """
    遍历所有txt文件，构建数据集。

    参数:
    data_dir (str): 数据文件夹路径
    fs (int): 采样率
    segment_length (int): 片段长度

    返回:
    EMG_Dataset: 完整数据集
    dict: 类别映射
    """
    all_segments = []
    all_labels = []
    class_mapping = {}

    # 获取所有txt文件
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

    for file in txt_files:
        file_path = os.path.join(data_dir, file)
        # 提取类别标签（文件名去掉.txt）
        class_name = file.replace('.txt', '').lower()
        if class_name not in class_mapping:
            class_mapping[class_name] = len(class_mapping)

        label = class_mapping[class_name]

        # 读取和处理信号
        raw_signal = read_emg_file(file_path)
        if len(raw_signal) < segment_length:  # 如果信号太短，跳过
            continue

        denoised_signal = denoise_signal(raw_signal, fs)
        segments = extract_active_segments(denoised_signal, fs, segment_length=segment_length)

        for segment in segments:
            all_segments.append(segment)
            all_labels.append(label)

    return EMG_Dataset(all_segments, all_labels), class_mapping

# 1D-CNN模型
class EMG_CNN(nn.Module):
    """
    轻量级1D-CNN模型
    """
    def __init__(self, num_classes):
        super(EMG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(2)

        # 计算全连接层输入大小 (假设输入长度1000)
        self.fc_input_size = 64 * (1000 // 8)  # 1000 / 2 / 2 / 2 = 125, 64*125=8000
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        features = x.view(x.size(0), -1)  # 展平为特征向量
        out = self.fc(features)
        return out, features

# 训练函数
def train_model(model, train_loader, test_loader, num_epochs=100, device='cpu'):
    """
    训练模型

    参数:
    model: 模型
    train_loader: 训练数据加载器
    test_loader: 测试数据加载器
    num_epochs (int): 训练轮数
    device: 设备

    返回:
    dict: 训练历史
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    return history

# 绘图函数
def plot_preprocessing_comparison(raw_signal, denoised_signal, sample_idx=0):
    """
    绘制信号预处理对比图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    time = np.arange(len(raw_signal)) / 1000  # 时间轴

    ax1.plot(time, raw_signal, color='blue', alpha=0.7)
    ax1.set_title('Original Signal', fontsize=14)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.grid(True)

    ax2.plot(time, denoised_signal, color='red', alpha=0.7)
    ax2.set_title('Denoised Signal', fontsize=14)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('out/preprocessing_comparison.png', dpi=600, bbox_inches='tight')
    # plt.show()

def plot_training_curves(history):
    """
    绘制训练过程曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'g-', label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('out/training_curves.png', dpi=600, bbox_inches='tight')
    # plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('out/confusion_matrix.png', dpi=600, bbox_inches='tight')
    # plt.show()

def plot_tsne(features, labels, class_names):
    """
    绘制t-SNE降维聚类图
    """
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   label=class_name, alpha=0.7, s=50)

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('out/tsne_visualization.png', dpi=600, bbox_inches='tight')
    # plt.show()

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 构建数据集
    print("正在构建数据集...")
    dataset, class_mapping = build_dataset()
    print(f"数据集大小: {len(dataset)}")
    print(f"类别映射: {class_mapping}")

    if len(dataset) == 0:
        print("没有找到有效数据，请检查数据文件。")
        return

    # 划分训练集和测试集 (8:2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    num_classes = len(class_mapping)
    model = EMG_CNN(num_classes)

    # 训练模型
    print("开始训练...")
    history = train_model(model, train_loader, test_loader, device=device)

    # 保存模型
    torch.save(model.state_dict(), 'emg_cnn_model.pth')
    print("模型已保存为 emg_cnn_model.pth")

    # 绘制图1：信号预处理对比
    print("绘制信号预处理对比图...")
    if len(dataset) > 0:
        sample_file = list(class_mapping.keys())[0] + '.txt'
        raw_signal = read_emg_file(os.path.join('data', sample_file))
        if len(raw_signal) > 0:
            denoised_signal = denoise_signal(raw_signal)
            plot_preprocessing_comparison(raw_signal, denoised_signal)

    # 绘制图2：训练曲线
    print("绘制训练曲线...")
    plot_training_curves(history)

    # 绘制图3：混淆矩阵
    print("绘制混淆矩阵...")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    class_names = list(class_mapping.keys())
    plot_confusion_matrix(y_true, y_pred, class_names)

    # 绘制图4：t-SNE
    print("绘制t-SNE可视化...")
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, features = model(inputs)
            features_list.extend(features.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    plot_tsne(np.array(features_list), np.array(labels_list), class_names)

    print("所有任务完成！")

if __name__ == "__main__":
    main()
