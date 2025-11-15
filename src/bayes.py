from __future__ import annotations

from typing import Tuple

import numpy as np


class GaussianNaiveBayes:
    """高斯朴素贝叶斯分类器。

    假设每个类别的特征服从独立的高斯分布。
    """

    def __init__(self) -> None:
        """初始化分类器。"""
        self.classes_: np.ndarray | None = None
        self.class_priors_: np.ndarray | None = None  # P(C_k)
        self.class_means_: np.ndarray | None = None   # 每个类别的均值向量 μ_k
        self.class_variances_: np.ndarray | None = None  # 每个类别的方差向量 σ_k²

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNaiveBayes:
        """在训练数据上拟合模型。

        Args:
            X: 训练特征矩阵，形状 (N, D)。
            y: 训练标签向量，形状 (N,)。

        Returns:
            GaussianNaiveBayes: 训练后的模型实例。
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 初始化参数
        self.class_priors_ = np.zeros(n_classes)
        self.class_means_ = np.zeros((n_classes, n_features))
        self.class_variances_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            # 选择属于类别 c 的样本
            X_c = X[y == c]

            # 计算先验概率 P(C_k)
            self.class_priors_[i] = len(X_c) / len(X)

            # 计算均值 μ_k
            self.class_means_[i] = np.mean(X_c, axis=0)

            # 计算方差 σ_k² (添加小常数避免除零)
            self.class_variances_[i] = np.var(X_c, axis=0, ddof=1) + 1e-9

        return self

    def _compute_likelihood(self, x: np.ndarray, class_idx: int) -> float:
        """计算样本 x 在给定类别下的似然概率 P(x|C_k)。

        Args:
            x: 单个样本，形状 (D,)。
            class_idx: 类别索引。

        Returns:
            float: 似然概率。
        """
        mean = self.class_means_[class_idx]
        var = self.class_variances_[class_idx]

        # 高斯概率密度函数 (忽略常数项 1/sqrt(2π))
        # P(x_i|C_k) = 1/sqrt(2π σ_{k,i}²) * exp(-0.5 * (x_i - μ_{k,i})² / σ_{k,i}²)
        # 我们只需要计算 exp(-0.5 * (x - μ)² / σ²) / sqrt(σ²)

        diff = x - mean
        exponent = -0.5 * diff ** 2 / var
        log_likelihood = np.sum(exponent - 0.5 * np.log(var))

        return np.exp(log_likelihood)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测样本的后验概率。

        Args:
            X: 测试特征矩阵，形状 (N, D)。

        Returns:
            np.ndarray: 后验概率矩阵，形状 (N, n_classes)。
        """
        if self.classes_ is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法。")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        posteriors = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for j in range(n_classes):
                # P(C_k|x) ∝ P(x|C_k) * P(C_k)
                likelihood = self._compute_likelihood(X[i], j)
                posteriors[i, j] = likelihood * self.class_priors_[j]

            # 归一化
            total = np.sum(posteriors[i])
            if total > 0:
                posteriors[i] /= total

        return posteriors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测样本的类别标签。

        Args:
            X: 测试特征矩阵，形状 (N, D)。

        Returns:
            np.ndarray: 预测标签向量，形状 (N,)。
        """
        posteriors = self.predict_proba(X)
        return self.classes_[np.argmax(posteriors, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算模型在测试数据上的准确率。

        Args:
            X: 测试特征矩阵，形状 (N, D)。
            y: 真实标签向量，形状 (N,)。

        Returns:
            float: 准确率。
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def evaluate_bayes_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """训练并评估贝叶斯分类器。

    Args:
        X_train: 训练特征矩阵。
        y_train: 训练标签向量。
        X_test: 测试特征矩阵。
        y_test: 真实标签向量。

    Returns:
        Tuple[float, np.ndarray]: (测试准确率, 混淆矩阵)。
    """
    # 训练模型
    clf = GaussianNaiveBayes()
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)

    # 计算混淆矩阵
    n_classes = len(np.unique(y_test))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[true, pred] += 1

    return accuracy, cm
