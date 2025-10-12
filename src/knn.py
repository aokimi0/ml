from __future__ import annotations

from typing import Tuple

import numpy as np


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
    tie_strategy: str = "min",
) -> int:
    """对单个样本进行 kNN 预测（欧氏距离，向量化实现）。

    Args:
        X_train: 训练特征矩阵，形状 (N, D)。
        y_train: 训练标签向量，形状 (N,)。
        x_query: 查询样本，形状 (D,)。
        k: 邻居个数。
        tie_strategy: 平票策略，"min" 取最小标签，"random" 随机选一个。

    Returns:
        int: 预测标签。
    """
    # 欧氏距离的平方即可，排序结果一致
    diff = X_train - x_query  # (N, D)
    dist2 = np.einsum("nd,nd->n", diff, diff)  # 高效计算 (N,)

    # 取前 k 个最近邻索引
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if k > len(X_train):
        raise ValueError("k 不能大于训练集样本数")

    nn_idx = np.argpartition(dist2, kth=k - 1)[:k]
    votes = y_train[nn_idx]

    # 投票
    num_classes = int(np.max(y_train)) + 1
    counts = np.bincount(votes, minlength=num_classes)
    winners = np.flatnonzero(counts == counts.max())

    if len(winners) == 1 or tie_strategy == "min":
        return int(winners.min())

    # 随机平票策略（需外部固定全局随机种子）
    return int(np.random.choice(winners))


def loo_eval(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    tie_strategy: str = "min",
) -> Tuple[float, np.ndarray]:
    """对整个数据集执行 LOO 评估。

    - 共 N 次，每次将第 i 个样本作为测试，其余为训练。
    - 采用欧氏距离（平方）衡量，避免数据泄漏：将自身距离设为 +inf。

    Args:
        X: 全量特征矩阵，形状 (N, D)。
        y: 全量标签向量，形状 (N,)。
        k: 邻居个数。
        tie_strategy: 平票策略，"min" 或 "random"。

    Returns:
        Tuple[float, np.ndarray]: (准确率, 混淆矩阵)。
    """
    N = X.shape[0]
    num_classes = int(np.max(y)) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)

    correct = 0
    for i in range(N):
        # 对单个查询样本向量化计算与所有样本的距离
        diff = X - X[i]  # (N, D)
        dist2 = np.einsum("nd,nd->n", diff, diff)
        dist2[i] = np.inf  # 自身设为无穷大，避免泄漏

        nn_idx = np.argpartition(dist2, kth=k - 1)[:k]
        votes = y[nn_idx]
        counts = np.bincount(votes, minlength=num_classes)
        winners = np.flatnonzero(counts == counts.max())
        if len(winners) == 1 or tie_strategy == "min":
            pred = int(winners.min())
        else:
            pred = int(np.random.choice(winners))

        cm[y[i], pred] += 1
        correct += int(pred == y[i])

    acc = correct / float(N)
    return acc, cm


