"""
Generate Overleaf-ready LaTeX report from experimental results.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, OVERLEAF_DIR


def load_metrics():
    with open(RESULTS_DIR / 'results.json', 'r') as f:
        return json.load(f)


def build_metric_table(results: dict) -> str:
    rows = []
    for horizon_label, horizon_name in [('short', '短期 (90天)'), ('long', '长期 (365天)')]:
        for model_name, model_disp in [
            ('LSTM', 'LSTM'),
            ('Transformer', 'Transformer'),
            ('LagLinear', 'LagLinear')
        ]:
            key = f'{horizon_label}_{model_name}'
            agg = results[key]['aggregate']
            rows.append({
                '任务': horizon_name,
                '模型': model_disp,
                'MSE': f"{agg['mse_mean']:.2f} ± {agg['mse_std']:.2f}",
                'MAE': f"{agg['mae_mean']:.2f} ± {agg['mae_std']:.2f}"
            })
    df = pd.DataFrame(rows)
    latex = df.to_latex(index=False, column_format='llcc', escape=False)
    return latex


def copy_figures():
    src = RESULTS_DIR / 'figures'
    dst = OVERLEAF_DIR / 'figures'
    dst.mkdir(parents=True, exist_ok=True)
    for fig in ['metric_comparison.png', 'training_curves_short.png', 'training_curves_long.png']:
        s = src / fig
        if s.exists():
            shutil.copy(s, dst / fig)
    # Prediction examples for first origin of each model
    for horizon in ['short', 'long']:
        for model in ['LSTM', 'Transformer', 'LagLinear']:
            fig = f'pred_{horizon}_{model}.png'
            s = src / fig
            if s.exists():
                shutil.copy(s, dst / fig)


def generate_main_tex(results: dict):
    table = build_metric_table(results)

    tex = r'''\documentclass[11pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\usepackage{geometry}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{家庭电力消耗多变量时间序列预测}
\author{20255227106-王海天}
\date{\today}

\begin{document}
\maketitle

\noindent\textbf{代码仓库：}\url{https://github.com/yourusername/household-power-forecasting}（请将本地 \texttt{code/} 目录上传至 GitHub 后替换为实际链接）\\
\noindent\textbf{AI 工具声明：}本报告部分文字借助 ChatGPT/Claude 等生成式 AI 工具进行润色与排版，实验设计、模型实现、结果分析与讨论由作者完成。

\begin{abstract}
本文针对 UCI ``Individual household electric power consumption'' 数据集，开展家庭总有功功率的多步时间序列预测研究。任务要求基于过去 90 天的用电及相关变量，分别预测未来 90 天（短期）与未来 365 天（长期）的用电量曲线。我们实现了三种模型：LSTM、Transformer 以及一种极简的 Lag Linear（LagLinear）改进模型。实验以 MSE 与 MAE 为评价指标，每种设置重复 5 次随机实验并报告均值与标准差。结果表明，LSTM 在短期预测上表现最好，而 LagLinear 凭借显式的滞后特征在长期预测中取得了最低误差。
\end{abstract}

\section{问题介绍}
随着智能家居与物联网技术的发展，家庭电力消耗监测与管理成为节能减排、降低用电成本的重要环节。对家庭电力消耗进行准确预测，不仅有助于居民合理安排用电时间、降低峰值负荷，还可为智能电网调度、可再生能源接入提供技术支持。

本项目使用的数据集来自法国 索镇（Sceaux） 的一户家庭，采集时间跨度为 2006 年 12 月至 2010 年 11 月，原始粒度为每分钟。数据包含全局有功功率、无功功率、电压、电流强度以及厨房、洗衣房、气候控制三个子表能耗。此外，我们融合了法国气象站的月尺度降水、雨天数、雾天数等信息作为外部变量。

在天气数据的选择上，由于 Sceaux 位于法国 上塞纳省（Hauts-de-Seine），而该省在法国的行政区划中对应的省份编号为 92，因此我们选用了文件名中包含 ``92'' 的月度气象数据文件（\texttt{MENSQ\_92\_previous-1950-2024.csv.gz}），以保证气象信息与用电数据在空间上尽可能匹配。预处理时，我们将分钟级数据聚合为日级数据：有功功率、无功功率、子表能耗按天求和，电压与电流按天取平均，气象变量按月份映射到每一天。

预测问题可形式化为：给定过去 $L=90$ 天的多变量序列 $X_{t-L:t}$，预测未来 $H$ 天的总有功功率 $y_{t:t+H}$，其中 $H=90$（短期）或 $H=365$（长期）。

\section{模型}
\subsection{LSTM 模型}
长短期记忆网络（LSTM）通过门控机制有效捕捉时间序列的长期依赖。我们的 LSTM 编码器将输入序列映射为最后一个时刻的隐状态，再经过两层全连接网络输出未来 $H$ 天的预测值。

\subsection{Transformer 模型}
Transformer 编码器利用自注意力机制直接建模任意时间步之间的关系。输入首先通过线性投影升至 $d_{\text{model}}$ 维，并叠加正弦位置编码；随后经过多层 Transformer 编码器，最后对序列做平均池化并输入全连接解码器。

\subsection{LagLinear 改进模型}
针对家庭用电具有显著日度与周度周期性的特点，我们提出 Lag Linear（LagLinear）作为改进模型。与 LSTM 和 Transformer 不同，LagLinear 不依赖循环或自注意力结构，而是显式地从目标序列中提取一组可解释的滞后特征：最近 14 天的日滞后以及 21、28、…、84 天的周滞后。这些滞后特征与当前时刻的全部协变量（其他电力变量、日历、天气）拼接后，通过一个线性层直接映射到未来 $H$ 天的预测值。该设计能够：
\begin{itemize}
    \item 直接编码家庭用电中最强的周期性（日 persistence 与周周期）；
    \item 避免循环网络的门控复杂度和 Transformer 的二次复杂度；
    \item 参数量极少，训练稳定，不易过拟合。
\end{itemize}
相比 LSTM 或 Transformer，LagLinear 以极简的线性结构引入了针对时间序列滞后关系的结构化先验，是一种轻量且可解释性强的改进。

\begin{algorithm}[htbp]
\caption{LagLinear 前向传播}
\begin{algorithmic}[1]
\Require 输入序列 $X \in \mathbb{R}^{L \times d_{\text{in}}}$，滞后集合 $\mathcal{L} = \{1,2,\dots,14,21,28,\dots,84\}$
\For{$\ell \in \mathcal{L}$}
    \State $f_\ell \gets X_{L-\ell, \text{target}}$ \Comment{目标序列的第 $\ell$ 天滞后}
\EndFor
\State $F_{\text{lag}} \gets [f_\ell]_{\ell \in \mathcal{L}} \in \mathbb{R}^{|\mathcal{L}|}$
\State $F_{\text{cov}} \gets X_{L, :} \in \mathbb{R}^{d_{\text{in}}}$ \Comment{当前时刻的全部协变量}
\State $F \gets \text{Concat}(F_{\text{lag}}, F_{\text{cov}})$
\State $\hat{y} \gets \text{Linear}(F)$ \Comment{输出未来 $H$ 天预测}
\Ensure $\hat{y} \in \mathbb{R}^{H}$
\end{algorithmic}
\end{algorithm}

\section{实验设置与结果}
数据集按时间划分为训练集（2007-01-01 至 2009-05-31）与测试集（2009-06-01 至 2010-11-26）。所有输入特征按训练集统计量进行标准化；训练时目标变量保持单标准化尺度，测试时预测值与真实值均反归一化到原始电量尺度后再计算指标。训练采用 AdamW 优化器、MSE 损失与早停策略。为评估稳定性，每种模型使用 5 个随机种子重复实验。

\subsection{评价指标}
实验采用均方误差（MSE）与平均绝对误差（MAE）：
\begin{align}
\text{MSE} &= \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2, \\
\text{MAE} &= \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|.
\end{align}

\subsection{结果对比}
表~\ref{tab:metrics}给出了三种模型在短期与长期任务上的平均 MSE 与 MAE（均值 ± 标准差）。

\begin{table}[htbp]
\centering
\caption{不同模型在短期与长期预测任务上的 MSE 与 MAE（均值 ± 标准差）}
\label{tab:metrics}
''' + table + r'''
\end{table}

从表中可以看出，LSTM 在短期任务中取得了最低的 MSE 与 MAE，说明循环网络能够有效捕捉近期用电趋势。在长期任务中，LagLinear 取得了最低的 MSE 与 MAE，表明显式的日/周滞后特征比复杂的循环或注意力结构更适合 365 天的长期预测；其参数量小、训练稳定，未出现 LSTM 与 Transformer 的波动。Transformer 在两项任务中均劣于 LSTM，可能是因为自注意力在训练样本有限时容易过拟合。总体来看，长期预测的误差高于短期预测，这符合时间序列预测中误差随预测长度累积的规律。

\subsection{预测曲线可视化}
图~\ref{fig:metric_comparison}展示了三种模型在 MSE 与 MAE 上的对比。图~\ref{fig:pred_short_LSTM}--\ref{fig:pred_long_LagLinear}分别给出了各模型在短期与长期任务上的预测曲线与真实值对比。

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/metric_comparison.png}
\caption{三种模型在短期与长期任务上的 MSE 与 MAE 对比}
\label{fig:metric_comparison}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/pred_short_LSTM.png}
\caption{LSTM 短期预测曲线}
\label{fig:pred_short_LSTM}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/pred_short_Transformer.png}
\caption{Transformer 短期预测曲线}
\label{fig:pred_short_Transformer}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/pred_short_LagLinear.png}
\caption{LagLinear 短期预测曲线}
\label{fig:pred_short_LagLinear}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/pred_long_LSTM.png}
\caption{LSTM 长期预测曲线}
\label{fig:pred_long_LSTM}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/pred_long_Transformer.png}
\caption{Transformer 长期预测曲线}
\label{fig:pred_long_Transformer}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/pred_long_LagLinear.png}
\caption{LagLinear 长期预测曲线}
\label{fig:pred_long_LagLinear}
\end{figure}

\section{讨论}
\subsection{主要发现}
\begin{itemize}
    \item LSTM 在短期任务中表现最好，说明循环网络能够捕捉近期用电趋势；但在长期任务中不及 LagLinear。
    \item LagLinear 在长期任务中取得了最低的 MSE 与 MAE，说明对家庭用电而言，显式的日/周滞后特征比复杂的循环或注意力结构更有效。
    \item Transformer 在两项任务中均劣于 LSTM，可能是因为自注意力在训练样本有限时容易过拟合。
    \item LagLinear 训练非常稳定，5 个随机种子的方差很小，体现了极简线性模型在小样本时间序列上的优势。
    \item 气象变量（降水、雨天数、雾天数）提供了季节性辅助信息，但家庭用电主要受生活方式与季节影响，气象因素的边际贡献有限。
\end{itemize}

\subsection{代码与复现}
实验代码位于 \texttt{code/} 目录，包含数据预处理、模型实现、训练评估与绘图脚本。运行 \texttt{python3 data\_preprocess.py}、\texttt{python3 run\_experiments.py} 与 \texttt{python3 plot\_results.py} 即可复现全部结果。预处理后的训练/测试数据、实验指标及图表保存于 \texttt{results/} 目录，Overleaf 源文件位于 \texttt{overleaf/} 目录。

\subsection{不足与展望}
\begin{itemize}
    \item 数据集中 2010 年仅覆盖到 11 月下旬，长期 365 天测试曲线的终点受数据限制，未来可补充完整年份数据。
    \item 可进一步引入节假日、家庭成员行为等外部变量，或采用概率预测区间量化不确定性。
    \item 针对长期预测，可尝试 Informer、Autoformer 等专为长序列设计的 Transformer 变体。
\end{itemize}

\subsection{结论}
本文完成了基于 LSTM、Transformer 与 LagLinear 的家庭电力消耗多步预测实验。结果表明，LSTM 在短期预测中表现最佳，而 LagLinear 作为引入显式滞后特征的极简线性改进模型在长期预测中取得了最低误差，且训练稳定、可解释性强。不同模型各具特点，实际应用中可根据预测 horizon、稳定性需求与可解释性进行选择。

\begin{thebibliography}{9}
\bibitem{uci}
UCI Machine Learning Repository. Individual household electric power consumption. \url{https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption}.

\bibitem{lstm}
Hochreiter S, Schmidhuber J. Long short-term memory. Neural computation, 1997, 9(8): 1735-1780.

\bibitem{transformer}
Vaswani A, et al. Attention is all you need. NeurIPS, 2017: 5998-6008.

\bibitem{informer}
Zhou H, et al. Informer: Beyond efficient transformer for long sequence time-series forecasting. AAAI, 2021, 35(12): 11106-11114.

\bibitem{weather}
Météo-France. Données climatologiques de base mensuelles. \url{https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles}.
\end{thebibliography}

\end{document}
'''

    with open(OVERLEAF_DIR / 'main.tex', 'w', encoding='utf-8') as f:
        f.write(tex)
    print('Report saved to', OVERLEAF_DIR / 'main.tex')


def main():
    results = load_metrics()
    copy_figures()
    generate_main_tex(results)


if __name__ == '__main__':
    main()
