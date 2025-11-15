#!/usr/bin/env python3
"""
机器学习实验二：回归分析

实现内容：
- 任务1：线性回归 – 最小二乘法（正规方程）
- 任务2：线性回归 - 梯度下降法
- 任务3：超参数调优 - 学习率分析
- 任务4：正则化 - 岭回归
- 拓展任务：模型选择 - 多项式回归

作者：aokimi
日期：2025-01-30
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 强制设置中文字体
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    # 设置matplotlib后端
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
except:
    pass

# 配置常量
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# 创建目录
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
log_filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-exp2-regression.log"
log_path = LOGS_DIR / log_filename
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LinearRegression:
    """线性回归基类"""

    def __init__(self):
        self.w = None
        self.b = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测函数"""
        return X @ self.w + self.b


def train_test_split_xy(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    """
    将一维特征数据分割为训练集和测试集

    Args:
        x: 特征数组
        y: 标签数组
        test_ratio: 测试集比例

    Returns:
        训练特征、训练标签、测试特征、测试标签
    """
    n = x.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return x[idx[test_size:]], y[idx[test_size:]], x[idx[:test_size]], y[idx[:test_size]]


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    """
    将多维特征数据分割为训练集和测试集

    Args:
        X: 特征矩阵
        y: 标签数组
        test_ratio: 测试集比例

    Returns:
        训练特征、训练标签、测试特征、测试标签
    """
    n = X.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return X[idx[test_size:]], y[idx[test_size:]], X[idx[:test_size]], y[idx[:test_size]]


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """
    对特征进行标准化处理

    Args:
        X_train: 训练特征
        X_test: 测试特征

    Returns:
        标准化后的训练特征和测试特征
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        均方误差
    """
    return float(np.mean((y_true - y_pred) ** 2))


def normal_equation_fit(x_train: np.ndarray, y_train: np.ndarray):
    """
    使用正规方程求解线性回归参数

    正规方程：θ = (X^T X)^(-1) X^T y
    对于单特征线性回归：y = w*x + b，构造设计矩阵Xb = [x, 1]

    Args:
        x_train: 训练特征（一维）
        y_train: 训练标签

    Returns:
        w: 权重参数
        b: 偏置参数
    """
    # 构造设计矩阵 [x, 1]
    Xb = np.column_stack([x_train, np.ones(len(x_train))])

    # 正规方程求解
    theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y_train

    w, b = theta[0], theta[1]
    logger.info(f"正规方程求解完成：w={w:.4f}, b={b:.4f}")
    return w, b


def batch_gradient_descent(X_train: np.ndarray, y_train: np.ndarray,
                           lr: float = 0.01, epochs: int = 200) -> tuple:
    """
    批量梯度下降算法实现线性回归

    Args:
        X_train: 训练特征矩阵 (n_samples, n_features)
        y_train: 训练标签 (n_samples,)
        lr: 学习率
        epochs: 训练轮数

    Returns:
        w: 权重参数
        b: 偏置参数
        hist: MSE历史记录列表
    """
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)  # 初始化权重
    b = 0.0  # 初始化偏置

    hist = []  # 记录每轮训练的MSE

    for epoch in range(epochs):
        # 前向传播：计算预测值
        y_pred = X_train @ w + b

        # 计算梯度
        dw = (2 / n_samples) * X_train.T @ (y_pred - y_train)
        db = (2 / n_samples) * np.sum(y_pred - y_train)

        # 更新参数
        w -= lr * dw
        b -= lr * db

        # 记录当前MSE
        current_mse = mse(y_train, y_pred)
        hist.append(current_mse)

        # 每20轮输出一次训练状态
        if (epoch + 1) % 20 == 0:
            logger.debug(f"Epoch {epoch+1:3d}: MSE={current_mse:.4f}, w={w}, b={b:.4f}")

    logger.info(f"梯度下降训练完成：{epochs}轮，学习率={lr}")
    logger.info(f"最终参数：w={w}, b={b:.4f}")
    return w, b, hist


def task2_gradient_descent():
    """
    任务2：线性回归 - 梯度下降法

    使用winequality-white.csv数据集，实现批量梯度下降算法训练线性回归模型，
    记录训练过程的MSE收敛曲线，计算训练和测试MSE。
    """
    logger.info("=== 开始任务2：梯度下降线性回归 ===")

    # 数据路径
    csv_path = DATA_DIR / "winequality-white.csv"
    if not csv_path.exists():
        logger.error(f"数据文件不存在：{csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    logger.info(f"数据集大小：{len(X)}个样本，特征数：{X.shape[1]}")
    logger.info(f"标签范围：y∈[{y.min():.2f}, {y.max():.2f}]")

    # 按4:1比例分割训练集和测试集
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    logger.info(f"训练集大小：{len(X_train)}，测试集大小：{len(X_test)}")

    # 特征标准化
    X_train_norm, X_test_norm = normalize(X_train, X_test)

    # 设置训练参数
    learning_rate = 0.05
    epochs = 300

    logger.info(f"训练参数：学习率={learning_rate}，训练轮数={epochs}")

    # 训练模型
    w, b, hist = batch_gradient_descent(X_train_norm, y_train, lr=learning_rate, epochs=epochs)

    # 计算最终误差
    y_train_pred = X_train_norm @ w + b
    y_test_pred = X_test_norm @ w + b
    train_mse_final = mse(y_train, y_train_pred)
    test_mse_final = mse(y_test, y_test_pred)

    logger.info(f"最终训练MSE：{train_mse_final:.4f}")
    logger.info(f"最终测试MSE：{test_mse_final:.4f}")

    # 绘制MSE收敛曲线
    plt.figure(figsize=(10, 6))
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(1, len(hist) + 1), hist, color='#1f77b4', linewidth=2, label='训练MSE')
    plt.xlabel('训练轮数 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    plt.title('任务2：批量梯度下降收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    fig_path = FIGURES_DIR / "task2_gd_convergence.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"MSE收敛曲线已保存至：{fig_path}")

    logger.info("=== 任务2完成 ===\n")


def task3_learning_rate_analysis():
    """
    任务3：超参数调优 - 学习率分析

    基于任务2的代码，尝试几个不同的学习率，绘制训练过程的MSE收敛曲线，
    分析最佳学习率以及学习率过大或过小对收敛过程的影响。
    """
    logger.info("=== 开始任务3：学习率分析 ===")

    # 数据路径
    csv_path = DATA_DIR / "winequality-white.csv"
    if not csv_path.exists():
        logger.error(f"数据文件不存在：{csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    logger.info(f"数据集大小：{len(X)}个样本，特征数：{X.shape[1]}")

    # 按4:1比例分割训练集和测试集
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    logger.info(f"训练集大小：{len(X_train)}，测试集大小：{len(X_test)}")

    # 特征标准化
    X_train_norm, X_test_norm = normalize(X_train, X_test)

    # 设置不同的学习率进行对比
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    epochs = 300

    logger.info(f"对比学习率：{learning_rates}")
    logger.info(f"训练轮数：{epochs}")

    # 存储每个学习率的结果
    results = {}

    for lr in learning_rates:
        logger.info(f"训练学习率：{lr}")

        # 训练模型
        w, b, hist = batch_gradient_descent(X_train_norm, y_train, lr=lr, epochs=epochs)

        # 计算最终误差
        y_train_pred = X_train_norm @ w + b
        y_test_pred = X_test_norm @ w + b
        train_mse_final = mse(y_train, y_train_pred)
        test_mse_final = mse(y_test, y_test_pred)

        results[lr] = {
            'hist': hist,
            'train_mse': train_mse_final,
            'test_mse': test_mse_final,
            'w': w,
            'b': b
        }

        logger.info(f"  lr={lr:.3f} -> 训练MSE={train_mse_final:.4f}, 测试MSE={test_mse_final:.4f}")

    # 绘制多条收敛曲线对比
    plt.figure(figsize=(12, 8))
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, lr in enumerate(learning_rates):
        hist = results[lr]['hist']
        plt.plot(range(1, len(hist) + 1), hist,
                color=colors[i % len(colors)],
                linewidth=2,
                label='.3f')

    plt.xlabel('训练轮数 (Epoch)')
    plt.ylabel('训练均方误差 (MSE)')
    plt.title('任务3：不同学习率下的MSE收敛曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    fig_path = FIGURES_DIR / "task3_lr_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"学习率对比曲线已保存至：{fig_path}")

    # 分析结果
    logger.info("学习率分析结果：")
    best_lr = min(results.keys(), key=lambda x: results[x]['test_mse'])
    logger.info(f"最佳学习率：{best_lr:.3f} (测试MSE: {results[best_lr]['test_mse']:.4f})")

    # 分析学习率影响
    logger.info("学习率影响分析：")
    logger.info("1. 学习率过小（如0.001）：收敛缓慢，需要更多训练轮数才能达到较好效果")
    logger.info("2. 学习率适中（如0.01-0.05）：收敛稳定，训练效率高，泛化性能好")
    logger.info("3. 学习率过大（如0.1-0.2）：可能导致震荡或发散，无法收敛到最优解")

    # 输出最终结果表格
    logger.info("各学习率最终结果：")
    logger.info("<8")
    logger.info("-" * 50)
    for lr in learning_rates:
        result = results[lr]
        converged = "是" if result['hist'][-1] < result['hist'][0] * 0.1 else "否"
        logger.info("<8")

    logger.info("=== 任务3完成 ===\n")


def ridge_regression_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    岭回归闭式解实现

    岭回归公式：θ = (X^T X + λI)^(-1) X^T y
    其中λ是正则化参数，I是单位矩阵

    Args:
        X: 特征矩阵 (n_samples, n_features)，已包含偏置列（常数1）
        y: 标签数组 (n_samples,)
        lam: 正则化参数 λ

    Returns:
        theta: 参数向量，包括权重和偏置
    """
    n_samples, n_features = X.shape

    # 构造正则化项 λI
    regularization = lam * np.eye(n_features)

    # 岭回归闭式解
    # 注意：为了数值稳定性，我们对X^T X的主对角线添加λ，而不是整个矩阵
    # 这是因为偏置项（最后一列）通常不进行正则化
    XTX = X.T @ X
    XTX[:-1, :-1] += regularization[:-1, :-1]  # 只对权重参数正则化

    theta = np.linalg.inv(XTX) @ X.T @ y

    logger.info(f"岭回归求解完成：λ={lam}, 参数维度={theta.shape}")
    return theta


def task4_ridge_regression():
    """
    任务4：正则化 - 岭回归

    使用winequality-white.csv数据集，实现岭回归的闭式解，
    设置正则化参数并计算训练和测试误差。
    """
    logger.info("=== 开始任务4：岭回归正则化 ===")

    # 数据路径
    csv_path = DATA_DIR / "winequality-white.csv"
    if not csv_path.exists():
        logger.error(f"数据文件不存在：{csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    logger.info(f"数据集大小：{len(X)}个样本，特征数：{X.shape[1]}")

    # 按4:1比例分割训练集和测试集
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    logger.info(f"训练集大小：{len(X_train)}，测试集大小：{len(X_test)}")

    # 特征标准化
    X_train_norm, X_test_norm = normalize(X_train, X_test)

    # 追加偏置列（常数1）
    X_train_ext = np.column_stack([X_train_norm, np.ones(X_train_norm.shape[0])])
    X_test_ext = np.column_stack([X_test_norm, np.ones(X_test_norm.shape[0])])

    # 设置正则化参数进行测试
    lambda_values = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

    logger.info(f"测试正则化参数：{lambda_values}")

    results = {}

    for lam in lambda_values:
        # 训练岭回归模型
        theta = ridge_regression_closed_form(X_train_ext, y_train, lam)

        # 分离权重和偏置
        w, b = theta[:-1], theta[-1]

        # 计算预测和误差
        y_train_pred = X_train_ext @ theta
        y_test_pred = X_test_ext @ theta
        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)

        # 计算参数范数（用于观察正则化效果）
        w_norm = np.linalg.norm(w)

        results[lam] = {
            'w': w,
            'b': b,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'w_norm': w_norm
        }

        logger.info(f"λ={lam:.3f} -> 训练MSE={train_mse:.4f}, 测试MSE={test_mse:.4f}, ||w||={w_norm:.4f}")

    # 绘制不同λ下的MSE对比
    plt.figure(figsize=(12, 5))
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 子图1：MSE对比
    plt.subplot(1, 2, 1)
    lambdas = list(results.keys())
    train_mses = [results[lam]['train_mse'] for lam in lambdas]
    test_mses = [results[lam]['test_mse'] for lam in lambdas]

    plt.plot(lambdas, train_mses, 'o-', label='训练MSE', color='#1f77b4', linewidth=2)
    plt.plot(lambdas, test_mses, 's-', label='测试MSE', color='#ff7f0e', linewidth=2)
    plt.xscale('log')
    plt.xlabel('正则化参数 λ')
    plt.ylabel('均方误差 (MSE)')
    plt.title('不同λ下的MSE对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：权重范数对比
    plt.subplot(1, 2, 2)
    w_norms = [results[lam]['w_norm'] for lam in lambdas]
    plt.plot(lambdas, w_norms, 'o-', label='||w||', color='#2ca02c', linewidth=2)
    plt.xscale('log')
    plt.xlabel('正则化参数 λ')
    plt.ylabel('权重范数 ||w||')
    plt.title('正则化参数对权重的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    fig_path = FIGURES_DIR / "task4_ridge_regression.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"岭回归结果图像已保存至：{fig_path}")

    # 分析结果
    best_lam = min(results.keys(), key=lambda x: results[x]['test_mse'])
    logger.info(f"最佳正则化参数：λ={best_lam:.3f} (测试MSE: {results[best_lam]['test_mse']:.4f})")

    # 输出分析
    logger.info("岭回归分析结果：")
    logger.info("1. 当λ=0时：等价于普通最小二乘法，可能出现过拟合")
    logger.info("2. 当λ增大时：权重范数减小，正则化效果增强，可能出现欠拟合")
    logger.info("3. 合适的λ值可以在偏差和方差之间取得平衡")

    # 输出详细结果表格
    logger.info("各正则化参数结果：")
    logger.info("<8")
    logger.info("-" * 60)
    for lam in lambda_values:
        result = results[lam]
        logger.info("<8")

    logger.info("=== 任务4完成 ===\n")


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """
    生成多项式特征

    Args:
        x: 输入特征数组 (n_samples,)
        degree: 多项式阶数

    Returns:
        多项式特征矩阵 (n_samples, degree+1)，包含 [1, x, x^2, ..., x^degree]
    """
    if degree < 0:
        raise ValueError("多项式阶数必须大于等于0")

    # 生成多项式特征： [1, x, x^2, ..., x^degree]
    features = []
    for d in range(degree + 1):
        features.append(x ** d)

    return np.column_stack(features)


def fit_polynomial_regression(x_train: np.ndarray, y_train: np.ndarray,
                             degree: int) -> np.ndarray:
    """
    使用闭式解训练多项式回归模型

    Args:
        x_train: 训练特征 (n_samples,)
        y_train: 训练标签 (n_samples,)
        degree: 多项式阶数

    Returns:
        theta: 多项式系数向量
    """
    # 生成多项式特征
    X_poly = polynomial_features(x_train, degree)

    # 使用正规方程求解
    theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y_train

    return theta


def predict_polynomial(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    多项式回归预测

    Args:
        x: 输入特征
        theta: 多项式系数

    Returns:
        预测值
    """
    degree = len(theta) - 1
    X_poly = polynomial_features(x, degree)
    return X_poly @ theta


def task_ext_polynomial_regression():
    """
    拓展任务：模型选择 - 多项式回归

    使用dataset_regression.csv数据集，实现多项式回归，
    尝试不同阶数并分析过拟合现象。
    """
    logger.info("=== 开始拓展任务：多项式回归 ===")

    # 数据路径
    csv_path = DATA_DIR / "dataset_regression.csv"
    if not csv_path.exists():
        logger.error(f"数据文件不存在：{csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=[np.number])
    x = num_df.iloc[:, 0].to_numpy().astype(float)
    y = num_df.iloc[:, 1].to_numpy().astype(float)

    logger.info(f"数据集大小：{len(x)}个样本")
    logger.info(f"特征范围：x∈[{x.min():.2f}, {x.max():.2f}], y∈[{y.min():.2f}, {y.max():.2f}]")

    # 数据分割
    x_train, y_train, x_test, y_test = train_test_split_xy(x, y, test_ratio=0.2)
    logger.info(f"训练集大小：{len(x_train)}，测试集大小：{len(x_test)}")

    # 特征标准化（用于多项式回归的数值稳定性）
    mean_x, std_x = x_train.mean(), x_train.std() + 1e-8
    x_train_std = (x_train - mean_x) / std_x
    x_test_std = (x_test - mean_x) / std_x

    # 设置不同的多项式阶数
    degrees = [1, 2, 4]

    logger.info(f"测试多项式阶数：{degrees}")

    results = {}

    for degree in degrees:
        logger.info(f"训练{degree}阶多项式回归")

        # 训练模型
        theta = fit_polynomial_regression(x_train_std, y_train, degree)

        # 预测
        y_train_pred = predict_polynomial(x_train_std, theta)
        y_test_pred = predict_polynomial(x_test_std, theta)

        # 计算误差
        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)

        results[degree] = {
            'theta': theta,
            'train_mse': train_mse,
            'test_mse': test_mse
        }

        logger.info(f"  {degree}阶多项式 -> 训练MSE={train_mse:.4f}, 测试MSE={test_mse:.4f}")
        logger.info(f"    系数：{theta}")

    # 绘制不同阶数多项式拟合结果
    plt.figure(figsize=(12, 8))
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制原始数据点
    plt.scatter(x_train, y_train, s=20, color='#1f77b4', alpha=0.6, label='训练数据')
    plt.scatter(x_test, y_test, s=20, color='#ff7f0e', alpha=0.6, label='测试数据')

    # 生成用于绘图的x值
    x_plot = np.linspace(x.min(), x.max(), 400)
    x_plot_std = (x_plot - mean_x) / std_x

    colors = ['#d62728', '#2ca02c', '#9467bd']
    for i, degree in enumerate(degrees):
        theta = results[degree]['theta']
        y_plot = predict_polynomial(x_plot_std, theta)
        train_mse = results[degree]['train_mse']
        test_mse = results[degree]['test_mse']

        plt.plot(x_plot, y_plot, color=colors[i % len(colors)], linewidth=2,
                label=f'{degree}阶多项式 (训练MSE={train_mse:.3f}, 测试MSE={test_mse:.3f})')

    plt.xlabel('特征 x')
    plt.ylabel('目标值 y')
    plt.title('拓展任务：不同阶数多项式回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    fig_path = FIGURES_DIR / "task_ext_polynomial_regression.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"多项式回归结果图像已保存至：{fig_path}")

    # 分析结果
    best_degree = min(results.keys(), key=lambda x: results[x]['test_mse'])
    logger.info(f"最佳多项式阶数：{best_degree} (测试MSE: {results[best_degree]['test_mse']:.4f})")

    # 分析过拟合现象
    logger.info("多项式回归分析结果：")
    logger.info("1. 1阶多项式（线性）：简单模型，训练和测试误差相近，可能欠拟合")
    logger.info("2. 2阶多项式：平衡复杂度，训练和测试误差都较小")
    logger.info("3. 4阶多项式：高阶多项式，训练误差很小但测试误差可能较大，出现过拟合")

    # 计算训练误差和测试误差的差距（用于检测过拟合）
    for degree in degrees:
        train_mse = results[degree]['train_mse']
        test_mse = results[degree]['test_mse']
        gap = test_mse - train_mse
        overfitting = "是" if gap > 0.1 else "否"
        logger.info(f"  {degree}阶多项式：训练误差={train_mse:.4f}, 测试误差={test_mse:.4f}, 误差差距={gap:.4f}, 是否过拟合：{overfitting}")

    logger.info("=== 拓展任务完成 ===\n")


def task1_normal_equation():
    """
    任务1：线性回归 – 最小二乘法（正规方程）

    使用dataset_regression.csv数据集，实现正规方程求解线性回归，
    构造5个测试样本进行预测，绘制散点图和拟合直线。
    """
    logger.info("=== 开始任务1：正规方程线性回归 ===")

    # 数据路径
    csv_path = DATA_DIR / "dataset_regression.csv"
    if not csv_path.exists():
        logger.error(f"数据文件不存在：{csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=[np.number])
    x = num_df.iloc[:, 0].to_numpy().astype(float)
    y = num_df.iloc[:, 1].to_numpy().astype(float)

    logger.info(f"数据集大小：{len(x)}个样本")
    logger.info(f"特征范围：x∈[{x.min():.2f}, {x.max():.2f}], y∈[{y.min():.2f}, {y.max():.2f}]")

    # 数据分割
    x_train, y_train, x_test, y_test = train_test_split_xy(x, y, test_ratio=0.2)
    logger.info(f"训练集大小：{len(x_train)}，测试集大小：{len(x_test)}")

    # 训练模型
    w, b = normal_equation_fit(x_train, y_train)

    # 预测
    y_train_pred = w * x_train + b
    y_test_pred = w * x_test + b

    # 计算误差
    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)

    logger.info(f"训练MSE：{train_mse:.4f}")
    logger.info(f"测试MSE：{test_mse:.4f}")

    # 构造5个新测试样本进行预测
    x_new = np.array([-12, -3, 0, 4.5, 11])
    y_new_pred = w * x_new + b

    logger.info("5个新样本预测结果：")
    for xi, yi in zip(x_new, y_new_pred):
        logger.info(f"  x={xi:.2f} -> y_pred={yi:.3f}")

    # 绘制结果
    plt.figure(figsize=(10, 6))
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 训练数据散点图
    plt.scatter(x_train, y_train, s=12, color='#1f77b4', label='训练数据')

    # 测试数据散点图
    plt.scatter(x_test, y_test, s=12, color='#ff7f0e', label='测试数据')

    # 拟合直线
    x_range = np.linspace(np.min(x), np.max(x), 200)
    y_fit = w * x_range + b
    plt.plot(x_range, y_fit, color='#d62728', linewidth=2,
             label='.3f')

    # 新样本预测点
    plt.scatter(x_new, y_new_pred, s=50, color='#2ca02c', marker='x',
                label='新样本预测', linewidth=2)

    plt.xlabel('特征 x')
    plt.ylabel('目标值 y')
    plt.title('任务1：正规方程线性回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    fig_path = FIGURES_DIR / "task1_normal_equation_fit.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"结果图像已保存至：{fig_path}")

    # 输出思考题答案
    logger.info("思考题答案：")
    logger.info("1. 正规方程的求解核心是计算矩阵逆(X^T X)^(-1)")
    logger.info("2. 当X^T X矩阵接近奇异（行列式接近0）时会失败，可通过添加正则化项或使用伪逆解决")

    logger.info("=== 任务1完成 ===\n")


if __name__ == '__main__':
    # 运行所有任务
    task1_normal_equation()
    task2_gradient_descent()
    task3_learning_rate_analysis()
    task4_ridge_regression()
    task_ext_polynomial_regression()
