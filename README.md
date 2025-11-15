## 项目简介

本项目包含两个机器学习实验：

1. **实验一**：基于 kNN 的手写数字识别（Semeion 数据集）与留一法（LOO）评估实验
2. **实验二**：回归分析（正规方程、梯度下降、岭回归、多项式回归）

代码遵循仓库规则与可复现性最佳实践，统一输出日志与图表，支持与 Weka IBk 结果对比。

## 环境要求

- WSL Ubuntu 24.04（推荐）
- Python 3.10+，依赖通过 Poetry 管理
- 仅 CPU 依赖（无需 GPU / CUDA）
- 已在 `pyproject.toml` 配置清华 PyPI 源

### 安装步骤

```bash
# 1) 安装依赖
poetry install

# 2) 可选：仅 CPU 版 Torch（本项目非必需）
poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

## 数据准备

### 实验一数据
- 下载并放置原始文件至 `data/semeion.data.txt`
  - 数据来源：`http://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit`
  - 格式：前 256 列为 16×16 像素（0/1），后 10 列为独热编码标签；标签通过 `argmax` 转为 0–9。

### 实验二数据
- `data/dataset_regression.csv`：线性回归数据集（自动从实验框架复制）
- `data/winequality-white.csv`：葡萄酒质量数据集（自动下载）

## 快速开始

### 实验一：kNN + LOO

```bash
# 运行实验（默认 k ∈ {1,3,5}）
poetry run python -m src.exp_1 --data data/semeion.data.txt --save-json
```

运行后：
- 终端打印各 k 的 LOO 准确率（保留 4 位小数）
- 日志文件：`logs/{YYYYMMDD-HHMMSS}-knn-loo.log`
- 混淆矩阵图：`reports/figures/knn-loo-k{K}.png`
- 结果汇总（可选）：`reports/knn_loo_results.json`

示例输出：

```text
k=1  LOO 准确率 = 0.9855
k=3  LOO 准确率 = 0.9868
k=5  LOO 准确率 = 0.9849
```

### 实验二：回归分析

```bash
# 运行所有回归实验任务
poetry run python src/exp_2.py
```

运行后：
- 终端打印各任务的详细结果和分析
- 日志文件：`logs/{YYYYMMDD-HHMMSS}-exp2-regression.log`
- 图表文件：
  - `reports/figures/task1_normal_equation_fit.png`：正规方程拟合结果
  - `reports/figures/task2_gd_convergence.png`：梯度下降收敛曲线
  - `reports/figures/task3_lr_comparison.png`：学习率对比分析
  - `reports/figures/task4_ridge_regression.png`：岭回归正则化分析
  - `reports/figures/task_ext_polynomial_regression.png`：多项式回归分析

## 命令行参数（核心）

来自 `src/exp_1.py`：

```text
--data       数据文件路径，默认 data/semeion.data.txt
--k          评估的 k 值列表，默认 1 3 5（示例：--k 1 3 5）
--seed       随机种子，默认 2025（影响平票随机策略）
--tie        平局策略：min|random，默认 min
--save-json  将结果写入 reports/knn_loo_results.json
```

示例：

```bash
poetry run python -m src.exp_1 --k 1 3 5 --tie min --seed 2025 --save-json
```

## 与 Weka 结果对比（可选）

脚本：`src/weka_eval.py` 会自动导出 ARFF 并调用 Weka IBk 做 LOO，对比精度并生成混淆矩阵。

```bash
# 在 WSL/Ubuntu 下建议先安装 weka（如需 sudo 权限）
sudo apt-get update -y && sudo apt-get install -y weka

# 运行 Weka LOO（默认 k ∈ {1,3,5}）
poetry run python -m src.weka_eval --data data/semeion.data.txt --save-json

# 若需禁用 Weka 的距离归一化（用于差异原因分析）
poetry run python -m src.weka_eval --no-norm --save-json
```

输出：
- 日志：`logs/{YYYYMMDD-HHMMSS}-weka-loo.log`
- 图表：`reports/figures/weka-knn-loo[-nonorm]-k{K}.png`
- JSON：`reports/weka_loo_results[-nonorm].json`

## 日志与可复现性

- 所有训练/评测日志统一写入 `logs/`（文件名：`{YYYYMMDD-HHMMSS}-{task}.log`）
- 固定随机种子（`random`、`numpy`）并在日志中记录参数：`k`、距离度量、平局策略
- 图表统一保存至 `reports/figures/`

## 目录结构

```text
.
├─ data/                     # 原始与中间数据（默认不入库）
├─ logs/                     # 训练与评测日志
├─ reports/
│  ├─ figures/               # 导出的图表（混淆矩阵等）
│  ├─ exp1_report.md|pdf     # 实验报告
│  └─ weka.jar|*.json        # Weka 相关/结果
├─ src/
│  ├─ exp_1.py               # 实验一入口（kNN + LOO）
│  ├─ exp_2.py               # 实验二入口（回归分析）
│  ├─ knn.py                 # kNN 与 LOO 实现
│  ├─ weka_eval.py           # Weka 对比评估脚本
│  └─ utils/
│     ├─ io_utils.py         # 日志、数据读取、JSON 工具
│     ├─ plot_utils.py       # 中文字体与混淆矩阵绘图
│     └─ arff_utils.py       # 导出 ARFF 文件
├─ lab1.md                   # 实验说明与要求
├─ pyproject.toml            # Poetry 依赖配置（含清华源）
└─ README.md
```

## 开发与提交规范

- 分支：`main`（稳定）/ `feat-xxx` / `fix-xxx`
- 提交信息：`<type>(scope): summary`，type ∈ {feat, fix, refactor, docs, chore, test}
- 合并前：自测通过、无 lint 错误，并在 PR 描述动机与影响面
- `.gitignore` 至少包含：`data/*`、`logs/*`、`.venv/`、`__pycache__/`、`.pytest_cache/`、`*.ipynb_checkpoints`

## 常见问题（FAQ）

- 找不到数据文件：请将原始 `semeion.data.txt` 放入 `data/` 目录；或将 train/test 拆分文件放根目录或 `data/`，程序会自动合并。
- 中文字体方块：首次运行会尝试使用系统中文字体；若缺失将自动下载并注册 `NotoSansSC` 到 `reports/fonts/`。
- Windows 原生环境：建议在 WSL Ubuntu 下运行，以避免字体/路径差异问题。
