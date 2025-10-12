from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def save_arff(X: np.ndarray, y: np.ndarray, out_path: Path | str, relation: str = "semeion") -> None:
    """将特征与标签导出为 ARFF 文件（class 为最后一列）。

    Args:
        X: 特征矩阵，形状 (N, 256)。
        y: 标签向量，形状 (N,)。应为 0..9 的整数标签。
        out_path: 输出文件路径。
        relation: ARFF relation 名称。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if X.ndim != 2 or X.shape[1] != 256:
        raise ValueError("X 维度应为 (N, 256)")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y 长度需与 X 行数一致，且为 1 维")

    classes = list(range(int(np.max(y)) + 1))
    with out_path.open("w", encoding="utf-8") as f:
        # 头部
        f.write(f"@relation {relation}\n\n")
        for i in range(256):
            f.write(f"@attribute pix{i:03d} NUMERIC\n")
        class_decl = ",".join(str(c) for c in classes)
        f.write(f"@attribute class {{{class_decl}}}\n\n")
        f.write("@data\n")

        # 数据（使用逗号分隔）
        # 将像素值压到 [0,1] 并输出为 0/1 或小数
        X_flat = X.astype(float)
        # 多数情况下是 0/1，这里直接保留浮点格式以兼容性最佳
        for row, label in zip(X_flat, y):
            values = ",".join(f"{v:.4f}" for v in row)
            f.write(f"{values},{int(label)}\n")


