# 家庭电力消耗多变量时间序列预测

本项目为 2026 年专硕机器学习课程项目代码，基于 UCI "Individual household electric power consumption" 数据集，对家庭总有功功率（global_active_power）进行多步预测。

## 任务

- **输入**：过去 90 天的多变量日级序列（电力、气象、日历特征）。
- **输出**：
  - 短期预测：未来 90 天
  - 长期预测：未来 365 天
- **模型**：
  1. LSTM
  2. Transformer
  3. LagLinear（改进模型）
- **评价指标**：MSE、MAE；进行 5 轮随机种子实验并报告均值与标准差。

## 数据说明

原始电力数据来自法国索镇（Sceaux）的一户家庭，时间粒度为每分钟。Sceaux 位于上塞纳省（Hauts-de-Seine），该省在法国的省份编号系统中为 **92**，因此天气数据选用 `MENSQ_92_previous-1950-2024.csv.gz`，以保证气象变量与用电数据在空间上匹配。

## 文件结构

```
.
├── README.md
├── requirements.txt
├── src/                    # 核心模块
│   ├── config.py           # 路径、超参数配置
│   ├── models.py           # LSTM / Transformer / LagLinear 模型
│   ├── dataset.py          # PyTorch Dataset 与滚动预测工具
│   └── train.py            # 训练、验证、评估函数
├── scripts/                # 可执行脚本
│   ├── data_preprocess.py  # 数据预处理
│   ├── run_experiments.py  # 主实验脚本
│   ├── plot_results.py     # 绘图脚本
│   └── run_all.sh          # 一键复现全部实验
└── experiments/            # 其他探索性脚本
    └── quick_test.py
```

## 运行环境

本项目依赖 Python 3.10+。推荐通过 pip 安装依赖：

```bash
pip install -r requirements.txt
```

`src/config.py` 会自动检测 GPU（`cuda`），若无 GPU 则回退到 CPU。

## 运行步骤

```bash
# 一键复现（推荐）
bash scripts/run_all.sh

# 或分步执行
python3 scripts/data_preprocess.py      # 生成 results/train.csv 与 results/test.csv
python3 scripts/run_experiments.py      # 训练并评估三种模型
python3 scripts/plot_results.py         # 生成结果图
```

## 输出

- `results/train.csv`, `results/test.csv`：预处理后的日级数据。
- `results/norm_params.json`：标准化参数。
- `results/results.json`, `results/results.pkl`：实验指标与预测结果。
- `results/figures/`：对比图、预测曲线图、训练曲线图。

