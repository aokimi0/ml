from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def ensure_dirs(paths: Iterable[Path | str]) -> None:
    """确保目录存在，不存在则创建。

    Args:
        paths: 需要创建的目录路径序列。
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    """固定随机种子，保证可复现。

    Args:
        seed: 随机种子整数。
    """
    random.seed(seed)
    np.random.seed(seed)


def _now_timestamp() -> str:
    """返回当前时间戳字符串，格式 YYYYMMDD-HHMMSS。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def init_logger(task_name: str) -> logging.Logger:
    """初始化日志记录器。

    所有日志写入 `logs/{YYYYMMDD-HHMMSS}-{task}.log`，同时输出到控制台。

    Args:
        task_name: 任务名，用于日志文件命名。

    Returns:
        logging.Logger: 配置完成的日志记录器实例。
    """
    logs_dir = Path("logs")
    ensure_dirs([logs_dir])
    log_file = logs_dir / f"{_now_timestamp()}-{task_name}.log"

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免重复添加 handler
    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.info("日志初始化完成，文件: %s", log_file)
    return logger


def load_semeion_data(data_path: Path | str | None = None, logger: logging.Logger | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """读取 Semeion 数据集。

    优先读取 `data/semeion.data.txt`。若不存在，则尝试拼接 `semeion_train.txt` 与 `semeion_test.txt`。

    Args:
        data_path: 数据文件路径，默认 `data/semeion.data.txt`。
        logger: 可选日志记录器。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 特征矩阵 X (N, 256) 与标签向量 y (N,)。
    """
    root = Path(".").resolve()
    default_data = Path(data_path) if data_path else Path("data/semeion.data.txt")

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    if default_data.exists():
        raw = np.loadtxt(default_data)
        X = raw[:, :256]
        y = np.argmax(raw[:, 256:], axis=1)
        _log(f"从 {default_data} 读取数据，样本数={len(X)}")
        return X, y

    # 备选：根目录或 data/ 下的 train/test 拆分文件
    candidates = [
        (root / "semeion_train.txt", root / "semeion_test.txt"),
        (root / "data" / "semeion_train.txt", root / "data" / "semeion_test.txt"),
    ]
    for train_p, test_p in candidates:
        if train_p.exists() and test_p.exists():
            train = np.loadtxt(train_p)
            test = np.loadtxt(test_p)
            raw = np.vstack([train, test])
            X = raw[:, :256]
            y = np.argmax(raw[:, 256:], axis=1)
            _log(f"合并 {train_p.name} + {test_p.name}，样本数={len(X)}")
            return X, y

    raise FileNotFoundError(
        f"未找到数据文件：{default_data} 或 train/test 拆分文件。请将原始文件放至 data/ 目录。"
    )


def save_json(obj: dict, path: Path | str) -> None:
    """保存 JSON 文件（UTF-8）。

    Args:
        obj: 需要保存的字典对象。
        path: 目标路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


