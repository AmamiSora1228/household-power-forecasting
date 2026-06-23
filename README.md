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

原始电力数据来自法国 索镇（Sceaux） 的一户家庭，时间粒度为每分钟。Sceaux 位于上塞纳省（Hauts-de-Seine），该省在法国的省份编号系统中为 **92**，因此天气数据选用 `MENSQ_92_previous-1950-2024.csv.gz`，以保证气象变量与用电数据在空间上匹配。

## 文件结构

```
code/
├── config.py            # 路径、超参数配置
├── data_preprocess.py   # 数据预处理
├── dataset.py           # PyTorch Dataset 与滚动预测工具
├── models.py            # LSTM / Transformer / LagLinear 模型
├── train.py             # 训练、验证、评估函数
├── run_experiments.py   # 主实验脚本
├── plot_results.py      # 绘图脚本
├── generate_report.py   # 生成 Overleaf LaTeX 报告
├── requirements.txt
└── README.md
```

## 运行环境

推荐使用 `sora` conda 环境，该环境已安装支持 CUDA 的 PyTorch：

```bash
conda activate sora
cd /storage/lcm_lab/sora/lab/work/code
```

`config.py` 会自动检测 GPU（`cuda`），若无 GPU 则回退到 CPU。

## 运行步骤

```bash
python3 data_preprocess.py      # 生成 results/train.csv 与 results/test.csv
python3 run_experiments.py      # 训练并评估三种模型（GPU 上约 1 分钟）
python3 plot_results.py         # 生成结果图
python3 generate_report.py      # 生成 overleaf/main.tex
```

## 输出

- `results/train.csv`, `results/test.csv`：预处理后的日级数据。
- `results/norm_params.json`：标准化参数。
- `results/results.json`, `results/results.pkl`：实验指标与预测结果。
- `results/figures/`：对比图、预测曲线图、训练曲线图。
- `overleaf/main.tex` 与 `overleaf/figures/`：Overleaf 报告源文件。

## GitHub 提交说明

课程要求提交代码 GitHub 链接。本项目的代码已上传至：

\`\`\`
https://github.com/AmamiSora1228/household-power-forecasting
\`\`\`

## 参考文献

- UCI Machine Learning Repository: Individual household electric power consumption.
- Hochreiter S, Schmidhuber J. Long short-term memory. Neural computation, 1997.
- Vaswani A, et al. Attention is all you need. NeurIPS, 2017.
- Zhou H, et al. Informer: Beyond efficient transformer for long sequence time-series forecasting. AAAI, 2021.
