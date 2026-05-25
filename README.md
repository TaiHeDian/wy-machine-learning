# EMG分类

使用方法：

1. 选择一个数据集目录
2. 定义一个分类任务
3. 定义类别名称及其顺序
4. 定义文件名如何映射到标签
5. 定义预处理、训练流程以及 CNN 结构
6. 运行训练，并将所有结果导出到一个带时间戳的输出文件夹中

## 项目结构

```text
|-- config.yaml
|-- data/
|-- out/
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- models.py
|   |-- pipeline.py
|   |-- plotting.py
|   |-- preprocessing.py
|   |-- reporting.py
|   |-- training.py
|   `-- utils.py
|-- main.py
`-- requirements.txt
```

* 将 `data/` 替换为其他数据集，并修改 `config.yaml`中的 `task`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

使用项目环境运行：

```bash
python main.py
```

## 输出目录结构

每次运行都会生成一个带时间戳的文件夹：

```text
out/
`-- 20260524_153000/
    |-- config_resolved.yaml
    |-- model.pth
    |-- report.md
    |-- figures/
    |   |-- confusion_matrix.png
    |   |-- confusion_matrix.tif
    |   |-- preprocessing_overview.png
    |   |-- preprocessing_overview.tif
    |   |-- training_curves.png
    |   |-- training_curves.tif
    |   |-- tsne.png
    |   `-- tsne.tif
    `-- tables/
        |-- confusion_counts.csv
        |-- confusion_percent.csv
        |-- metrics_summary.csv
        |-- per_class_accuracy.csv
        |-- predictions.csv
        |-- signal_inventory.csv
        |-- training_history.csv
        `-- tsne_embeddings.csv
```

## 配置

所有行为均由 `config.yaml` 定义。

### 任务配置

```yaml
task:
  name: words
  title: Confusion Matrix
  labels: ["no", "thanks", "come", "arrive", "drink", "like", "delight"]
  label_source: stem
  label_map: {}
  confusion_color: "#2B6CB0"
```

* `labels` 定义混淆矩阵顺序、预测顺序以及表格顺序
* `label_source` 控制标签来源于文件名（stem）还是父文件夹名称
* `label_map` 允许在不修改代码的情况下，将任意文件名映射到类别名称

例如：如果数据集使用的是 `1.txt` 到 `5.txt` 这样的文件名，只需修改配置即可运行：

```yaml
task:
  name: numeric_demo
  title: Numeric Label Demo
  labels: ["class_a", "class_b", "class_c", "class_d", "class_e"]
  label_source: stem
  label_map:
    "1": "class_a"
    "2": "class_b"
    "3": "class_c"
    "4": "class_d"
    "5": "class_e"
```

### 数据解析

```yaml
data:
  data_dir: data
  file_glob: "*.txt"
  encodings: ["utf-8", "gbk", "latin1"]
  header_lines: 3
  sample_rate_regex: "(\\d+)\\s*Hz"
  value_column: 3
  delimiter_regex: "[\\s,\\t]+"
```

该部分定义了：

* 数据所在位置
* 如何查找文件
* 保留多少行文件头
* 如何从文件头中提取采样率
* 哪一列包含信号值
* 如何将每一行拆分为字段

例如，对于单列信号文件，修改 `value_column`为 `1`。

## 预处理

```yaml
preprocessing:
  segment_ms: 1000
  window_ms: 100
  threshold_factor: 2.0
  notch_hz: 50.0
  notch_q: 30.0
  bandpass_low_hz: 20.0
  bandpass_high_hz: 400.0
  augment: true
```

## 训练

```yaml
training:
  seed: 42
  test_size: 0.2
  batch_size: 32
  epochs: 80
  learning_rate: 0.0005
  weight_decay: 0.0005
  patience: 20
  scheduler_factor: 0.5
  scheduler_patience: 4
  num_workers: 0
  device: auto
  enable_early_stopping: true
```

`enable_early_stopping`默认为 `true`，设置训练时是否需要**早停**。

## CNN 结构

```yaml
model:
  channels: [32, 64, 128, 128]
  kernel_sizes: [7, 5, 5, 3]
  pool_sizes: [2, 2, 2, 1]
  adaptive_pool_size: 16
  embedding_dim: 128
  dropout: 0.3
  use_batch_norm: true
```

网络的深度与宽度可以通过配置文件进行修改。
