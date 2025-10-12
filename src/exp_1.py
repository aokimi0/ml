from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from .knn import loo_eval
from .utils.io_utils import ensure_dirs, init_logger, load_semeion_data, set_random_seed, save_json
from .utils.plot_utils import setup_chinese_font, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="kNN + LOO 实验入口（欧氏距离，k∈{1,3,5}，日志与图表输出）"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/semeion.data.txt",
        help="数据文件路径，默认 data/semeion.data.txt；若缺失将尝试合并 train/test。",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="评估的 k 值列表，默认 1 3 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="随机种子（影响平票随机策略与示例可复现）",
    )
    parser.add_argument(
        "--tie",
        type=str,
        choices=["min", "random"],
        default="min",
        help="投票平局策略：min=取最小标签，random=随机选择",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="是否将各 k 的结果导出为 reports/knn_loo_results.json",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：读取数据、执行 LOO，并输出日志与图表。"""
    args = parse_args()

    # 目录与日志
    ensure_dirs(["logs", "reports/figures", "reports/fonts", "data"])  # data 仅保障目录存在
    logger = init_logger("knn-loo")
    set_random_seed(args.seed)
    logger.info("参数配置 | data=%s | k=%s | seed=%d | tie=%s", args.data, args.k, args.seed, args.tie)

    # 字体配置（中文）
    setup_chinese_font(logger)

    # 读数
    X, y = load_semeion_data(Path(args.data), logger)
    logger.info("数据维度 | X=%s | y=%s | 类别数=%d", X.shape, y.shape, int(np.max(y)) + 1)

    results = {}
    classes = [str(i) for i in range(int(np.max(y)) + 1)]

    # 逐 k 评估
    for k in args.k:
        logger.info("开始 LOO 评估 | k=%d", k)
        acc, cm = loo_eval(X, y, k=k, tie_strategy=args.tie)
        logger.info("完成 | k=%d | LOO 准确率=%.4f", k, acc)
        print(f"k={k}  LOO 准确率 = {acc:.4f}")

        # 图表输出
        fig_path = Path("reports/figures") / f"knn-loo-k{k}.png"
        plot_confusion_matrix(cm, classes=classes, title=f"k={k} 的混淆矩阵", out_path=fig_path)
        logger.info("混淆矩阵图已保存: %s", fig_path)

        results[str(k)] = {"acc": round(float(acc), 4)}

    # 可选：保存指标 JSON
    if args.save_json:
        json_path = Path("reports/knn_loo_results.json")
        save_json(results, json_path)
        logger.info("结果 JSON 已保存: %s", json_path)


if __name__ == "__main__":
    main()


