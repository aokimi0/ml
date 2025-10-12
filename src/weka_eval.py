from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np

from .utils.io_utils import ensure_dirs, init_logger, load_semeion_data, set_random_seed
from .utils.arff_utils import save_arff
from .utils.plot_utils import setup_chinese_font, plot_confusion_matrix


def _glob_jars(dir_path: Path) -> list[str]:
    return [str(p) for p in sorted(dir_path.glob("*.jar"))]


def ensure_weka_classpath(out_dir: Path, logger) -> str:
    """构建可用的 Weka classpath。

    优先使用系统 `/usr/share/java/*.jar`（apt 安装 weka），否则尝试下载单 jar。

    Returns:
        str: Java -cp 的 classpath 字符串（以 : 连接）。
    """
    sys_java_dir = Path("/usr/share/java")
    if sys_java_dir.exists():
        jars = _glob_jars(sys_java_dir)
        if any(j.endswith("/weka.jar") for j in jars):
            logger.info("检测到系统 weka 安装，使用 /usr/share/java/*.jar")
            return ":".join(jars)

    # 尝试通过 apt 安装 weka
    try:
        logger.info("未检测到系统 weka，尝试通过 apt 安装 weka …")
        subprocess.run(["sudo", "apt-get", "update", "-y"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "weka"], check=True)
        jars = _glob_jars(sys_java_dir)
        if any(j.endswith("/weka.jar") for j in jars):
            logger.info("apt 安装 weka 成功，使用 /usr/share/java/*.jar")
            return ":".join(jars)
    except Exception as e:
        logger.info("apt 安装 weka 失败：%s", e)

    # 兜底：下载单 jar（可能缺依赖，不推荐）
    jar_path = out_dir / "weka.jar"
    if not jar_path.exists():
        urls = [
            "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar",
            "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-dev/3.9.6/weka-dev-3.9.6.jar",
        ]
        for url in urls:
            try:
                subprocess.run(["curl", "-L", "-o", str(jar_path), url], check=True)
                break
            except Exception:
                continue
    logger.info("使用下载的 weka.jar（可能缺少依赖）: %s", jar_path)
    return str(jar_path)


def run_weka_ibk_loocv(classpath: str, arff_path: Path, k: int, seed: int, no_norm: bool = False) -> tuple[float, np.ndarray]:
    """调用 Weka CLI（IBk）执行 LOO 评估并解析输出。

    Args:
        weka_jar: weka.jar 路径。
        arff_path: 数据 ARFF 文件路径，class 为最后一列。
        k: KNN 的 k 值。
        seed: 随机种子（尽管 LOO 不依赖，但保持一致性）。

    Returns:
        (acc, cm): 准确率与混淆矩阵。
    """
    # -x N 表示 N 折交叉验证；对于 LOO，N=样本数
    # 输出需要解析 “Correctly Classified Instances” 与 “Confusion Matrix” 段
    base = [
        "java", "-cp", classpath,
        "weka.classifiers.lazy.IBk",
        "-K", str(k),
    ]
    if no_norm:
        # 使用不归一化的欧氏距离
        base += [
            "-A",
            'weka.core.neighboursearch.LinearNNSearch -A "weka.core.EuclideanDistance -R first-last -D"',
        ]
    cmd = base + [
        "-x", "-1",  # 先占位，稍后替换为样本数
        "-t", str(arff_path),
        "-s", str(seed),
    ]

    # 先读取样本数
    with arff_path.open("r", encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip() and not line.startswith("@"))
    cmd[cmd.index("-x") + 1] = str(n)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    out = proc.stdout

    # 解析准确率
    acc = None
    for line in out.splitlines():
        if "Correctly Classified Instances" in line and "%" in line:
            # e.g. Correctly Classified Instances      1571               98.617  %
            try:
                percent = float(line.strip().split()[-2])
                acc = percent / 100.0
                break
            except Exception:
                continue
    if acc is None:
        raise RuntimeError("未能从 Weka 输出解析到准确率")

    # 解析混淆矩阵（Weka 会在输出底部提供一个矩阵）
    # 这里简单策略：收集最后一个方阵区域
    lines = out.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("=== Confusion Matrix ==="):
            start = i
    if start is None:
        # 某些版本只在 "Stratified cross-validation" 段后给出矩阵，简单跳过
        cm = None
    else:
        # 读取矩阵行（形如:  1520   1   0 ... |   0 = 0）
        rows: list[list[int]] = []
        for line in lines[start + 1 : start + 1 + 64]:
            if "|" not in line:
                continue
            left = line.split("|")[0].strip()
            if not left:
                continue
            try:
                vals = [int(x) for x in left.split()]
            except Exception:
                continue
            rows.append(vals)
        if rows:
            cm_arr = np.array(rows, dtype=int)
            side = min(cm_arr.shape[0], cm_arr.shape[1])
            cm = cm_arr[:side, :side]
        else:
            cm = None

    if cm is None:
        # 若无法解析矩阵，则返回零矩阵占位
        cm = np.zeros((10, 10), dtype=int)

    return acc, cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Weka CLI 进行 IBk + LOO 评估")
    parser.add_argument("--data", type=str, default="data/semeion.data.txt", help="数据文件路径")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], help="评估的 k 列表")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--save-json", action="store_true", help="是否导出 JSON 结果")
    parser.add_argument("--no-norm", action="store_true", help="关闭 Weka 欧氏距离归一化")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(["reports", "reports/figures"]) 
    logger = init_logger("weka-loo")
    set_random_seed(args.seed)
    setup_chinese_font(logger)

    # 加载并导出 ARFF
    X, y = load_semeion_data(args.data, logger)
    arff_path = Path("reports/semeion.arff")
    save_arff(X, y, arff_path)
    logger.info("ARFF 文件已生成: %s", arff_path)

    # 准备 weka.jar
    classpath = ensure_weka_classpath(Path("reports"), logger)
    logger.info("Weka classpath: %s", classpath)

    results = {}
    classes = [str(i) for i in range(int(np.max(y)) + 1)]

    suffix = "-nonorm" if args.no_norm else ""
    for k in args.k:
        logger.info("开始 Weka LOO | k=%d | no_norm=%s", k, args.no_norm)
        acc, cm = run_weka_ibk_loocv(classpath, arff_path, k, args.seed, no_norm=args.no_norm)
        logger.info("完成 Weka LOO | k=%d | acc=%.4f | no_norm=%s", k, acc, args.no_norm)
        print(f"[Weka] k={k}  LOO 准确率 = {acc:.4f}")
        fig_path = Path("reports/figures") / f"weka-knn-loo{suffix}-k{k}.png"
        plot_confusion_matrix(cm, classes, title=f"Weka IBk k={k} 混淆矩阵", out_path=fig_path)
        results[str(k)] = {"acc": round(float(acc), 4)}

    if args.save_json:
        out_json = Path(f"reports/weka_loo_results{suffix}.json")
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Weka 结果 JSON 已保存: %s", out_json)


if __name__ == "__main__":
    main()


