from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
import numpy as np


def setup_chinese_font(logger: logging.Logger | None = None) -> None:
    """配置 matplotlib 中文字体，自动下载 NotoSansSC（如缺失）。

    - 优先使用本机已安装的中文字体（如 Noto Sans SC / 思源黑体 / SimHei）。
    - 若均不可用，则下载 `NotoSansSC-Regular.otf` 至 `reports/fonts/` 并注册。

    Args:
        logger: 可选日志记录器。
    """
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    matplotlib.rcParams["axes.unicode_minus"] = False

    prefer_names = [
        "Noto Sans CJK SC",  # fonts-noto-cjk 提供的常用家族名
        "Noto Sans SC",
        "Source Han Sans SC",
        "思源黑体",
        "SimHei",
    ]

    # 检查是否已有可用字体
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in prefer_names:
        if any(name in fam for fam in installed):
            matplotlib.rcParams["font.sans-serif"] = [name]
            _log(f"使用已安装中文字体: {name}")
            return

    # 尝试扫描系统字体并注册（适配 fonts-noto-cjk 等路径）
    try:
        candidates = []
        for ext in ("ttf", "otf", "ttc"):
            candidates += fm.findSystemFonts(fontext=ext)
        # 过滤包含关键字的字体文件
        picked = [p for p in candidates if any(key.lower() in p.lower() for key in (
            "notosanscjk", "notosanscjksc", "notosans sc", "sourcehansans", "source-han-sans",
            "思源黑体", "simhei"
        ))]
        # 注册并读取真实家族名
        for p in picked:
            try:
                fm.fontManager.addfont(p)
                name = FontProperties(fname=p).get_name()
                if name:
                    matplotlib.rcParams["font.family"] = ["sans-serif"]
                    matplotlib.rcParams["font.sans-serif"] = [name]
                    _log(f"已注册并使用中文字体: {name} ({p})")
                    return
            except Exception:
                continue
        # 回退到候选列表，若仍失败则由系统默认字体处理
        matplotlib.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "思源黑体", "SimHei"
        ]
        _log("已设置中文字体候选列表，若系统存在对应字体将自动生效。")
        return
    except Exception:
        pass

    # 下载 NotoSansSC-Regular.otf 到 reports/fonts/
    fonts_dir = Path("reports/fonts")
    fonts_dir.mkdir(parents=True, exist_ok=True)
    font_path = fonts_dir / "NotoSansSC-Regular.otf"
    if not font_path.exists():
        urls = [
            # GitHub raw 主链接
            "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
            # 备用：github.com 的 raw 路径
            "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
        ]
        last_err = None
        for url in urls:
            try:
                _log(f"未检测到中文字体，尝试下载: {url}")
                urllib.request.urlretrieve(url, font_path)
                _log(f"字体下载完成: {font_path}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                _log(f"下载失败，尝试下一个源。错误: {e}")
        if last_err is not None:
            _log("所有字体下载源均失败，继续使用默认英文字体（中文可能显示为方块）。")
            return

    try:
        fm.fontManager.addfont(str(font_path))
        # 注册后，家族名可能识别为 "Noto Sans SC" 或 "NotoSansSC"
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans SC", "NotoSansSC"]
        _log(f"已注册中文字体: {font_path.name}")
    except Exception as e:
        _log(f"字体注册失败，继续使用默认字体。错误: {e}")


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    title: str,
    out_path: Path | str,
) -> None:
    """绘制并保存混淆矩阵热力图。

    Args:
        cm: 混淆矩阵 (num_classes, num_classes)。
        classes: 类别名称列表。
        title: 图标题。
        out_path: 输出文件路径。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="真实标签",
        xlabel="预测标签",
        title=title,
    )

    # 旋转 x 轴标签以便阅读
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 文本标注
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


