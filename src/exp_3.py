from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .bayes import evaluate_bayes_classifier
from .utils.io_utils import ensure_dirs, init_logger, set_random_seed, save_json
from .utils.plot_utils import setup_chinese_font, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="贝叶斯分类器实验（高斯朴素贝叶斯，两个数据集）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="是否将结果导出为 reports/bayes_results.json",
    )
    return parser.parse_args()


def generate_datasets(seed: int = 42) -> tuple:
    """生成实验数据集。

    Returns:
        tuple: (X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test)
    """
    # 设置随机种子
    np.random.seed(seed)

    # 生成数据集1：高分离度
    X1, y1 = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed
    )

    # 生成数据集2：低分离度
    X2, y2 = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=0.5,
        random_state=seed
    )

    # 划分训练集和测试集
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=seed)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=seed)

    return X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test


def plot_data_distribution(X1: np.ndarray, y1: np.ndarray, X2: np.ndarray, y2: np.ndarray) -> None:
    """绘制数据分布图。

    Args:
        X1: 数据集1特征矩阵。
        y1: 数据集1标签向量。
        X2: 数据集2特征矩阵。
        y2: 数据集2标签向量。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 数据集1
    axes[0].scatter(X1[y1==0, 0], X1[y1==0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
    axes[0].scatter(X1[y1==1, 0], X1[y1==1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
    axes[0].set_title('数据集1 (高分离度)')
    axes[0].set_xlabel('特征1')
    axes[0].set_ylabel('特征2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 数据集2
    axes[1].scatter(X2[y2==0, 0], X2[y2==0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
    axes[1].scatter(X2[y2==1, 0], X2[y2==1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
    axes[1].set_title('数据集2 (低分离度)')
    axes[1].set_xlabel('特征1')
    axes[1].set_ylabel('特征2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/bayes_data_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    print('数据分布图已保存: reports/figures/bayes_data_distribution.png')


def plot_decision_boundary(clf, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, dataset_name: str) -> None:
    """绘制决策边界。

    Args:
        clf: 训练好的分类器。
        X_train: 训练特征矩阵。
        y_train: 训练标签向量。
        X_test: 测试特征矩阵。
        y_test: 测试标签向量。
        dataset_name: 数据集名称。
    """
    # 创建网格点
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测网格点
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

    # 绘制训练数据点
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='red', label='训练集-类别0', alpha=0.6, edgecolors='k')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue', label='训练集-类别1', alpha=0.6, edgecolors='k')

    # 绘制测试数据点（用不同形状）
    plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='darkred', marker='x', s=80, label='测试集-类别0', alpha=0.8)
    plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='darkblue', marker='x', s=80, label='测试集-类别1', alpha=0.8)

    plt.title(f'{dataset_name} - 贝叶斯分类器决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'reports/figures/bayes_decision_boundary_{dataset_name.lower().replace(" ", "_")}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f'决策边界图已保存: reports/figures/bayes_decision_boundary_{dataset_name.lower().replace(" ", "_")}.png')


def main() -> None:
    """主流程：运行贝叶斯分类器实验。"""
    args = parse_args()

    # 目录与日志
    ensure_dirs(["logs", "reports/figures", "reports/fonts", "data"])
    logger = init_logger("bayes")
    set_random_seed(args.seed)
    logger.info("参数配置 | seed=%d", args.seed)

    # 字体配置（中文）
    setup_chinese_font(logger)

    print('='*60)
    print('实验四：贝叶斯分类器')
    print('='*60)

    # 生成数据
    print('\n生成数据集...')
    X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test = generate_datasets(args.seed)

    print(f'\n数据集1: 训练集{X1_train.shape[0]}样本, 测试集{X1_test.shape[0]}样本')
    print(f'数据集2: 训练集{X2_train.shape[0]}样本, 测试集{X2_test.shape[0]}样本')

    # 绘制数据分布
    plot_data_distribution(
        np.vstack([X1_train, X1_test]), np.hstack([y1_train, y1_test]),
        np.vstack([X2_train, X2_test]), np.hstack([y2_train, y2_test])
    )

    results = {}

    # 数据集1：高分离度
    print('\n' + '='*40)
    print('数据集1 (高分离度) 实验')
    print('='*40)

    logger.info("开始数据集1实验")
    acc1, cm1 = evaluate_bayes_classifier(X1_train, y1_train, X1_test, y1_test)
    error_rate1 = 1 - acc1

    logger.info("数据集1 | 测试准确率=%.4f | 错误率=%.4f", acc1, error_rate1)
    print(f"测试准确率 = {acc1:.4f}")
    print(f"错误率 = {error_rate1:.4f}")

    # 保存混淆矩阵图
    classes = ['类别0', '类别1']
    fig_path = Path("reports/figures") / "bayes_cm_dataset1.png"
    plot_confusion_matrix(cm1, classes=classes, title="数据集1 - 贝叶斯分类器混淆矩阵", out_path=fig_path)
    logger.info("混淆矩阵图已保存: %s", fig_path)

    # 绘制决策边界
    from .bayes import GaussianNaiveBayes
    clf1 = GaussianNaiveBayes()
    clf1.fit(X1_train, y1_train)
    plot_decision_boundary(clf1, X1_train, y1_train, X1_test, y1_test, "数据集1")

    results["dataset1"] = {
        "accuracy": round(float(acc1), 4),
        "error_rate": round(float(error_rate1), 4)
    }

    # 数据集2：低分离度
    print('\n' + '='*40)
    print('数据集2 (低分离度) 实验')
    print('='*40)

    logger.info("开始数据集2实验")
    acc2, cm2 = evaluate_bayes_classifier(X2_train, y2_train, X2_test, y2_test)
    error_rate2 = 1 - acc2

    logger.info("数据集2 | 测试准确率=%.4f | 错误率=%.4f", acc2, error_rate2)
    print(f"测试准确率 = {acc2:.4f}")
    print(f"错误率 = {error_rate2:.4f}")

    # 保存混淆矩阵图
    fig_path = Path("reports/figures") / "bayes_cm_dataset2.png"
    plot_confusion_matrix(cm2, classes=classes, title="数据集2 - 贝叶斯分类器混淆矩阵", out_path=fig_path)
    logger.info("混淆矩阵图已保存: %s", fig_path)

    # 绘制决策边界
    clf2 = GaussianNaiveBayes()
    clf2.fit(X2_train, y2_train)
    plot_decision_boundary(clf2, X2_train, y2_train, X2_test, y2_test, "数据集2")

    results["dataset2"] = {
        "accuracy": round(float(acc2), 4),
        "error_rate": round(float(error_rate2), 4)
    }

    # 可选：保存指标 JSON
    if args.save_json:
        json_path = Path("reports/bayes_results.json")
        save_json(results, json_path)
        logger.info("结果 JSON 已保存: %s", json_path)

    # 实验分析
    print('\n' + '='*60)
    print('实验结果分析')
    print('='*60)

    print(f'数据集1准确率: {acc1:.4f}, 错误率: {error_rate1:.4f}')
    print(f'数据集2准确率: {acc2:.4f}, 错误率: {error_rate2:.4f}')

    if acc1 > acc2:
        print(f"高分离度数据集表现更好 (准确率差值: {acc1 - acc2:.4f})")
    elif acc2 > acc1:
        print(f"低分离度数据集表现更好 (准确率差值: {acc2 - acc1:.4f})")
    else:
        print("两个数据集表现相同")

    print('\n贝叶斯分类器特点:')
    print('- 假设特征条件独立（朴素假设）')
    print('- 假设特征服从高斯分布')
    print('- 计算效率高，无需迭代优化')
    print('- 对高分离度数据表现更好')

    print('\n'+'='*60)
    print('实验完成！')
    print('='*60)


if __name__ == "__main__":
    main()
