"""报告图表统一样式(dataviz 方法:validated default palette,light mode)。

用法:
    from figstyle import apply_style, CAT, SEQ, DIV_POS, DIV_NEG, INK, MUTED
    apply_style()
"""
import matplotlib as mpl

# categorical 固定顺序(经 CVD 校验的顺序,不许循环重排)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# sequential(单色蓝,浅->深)
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"  # 蓝-灰-红
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "sans-serif"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 9,
        "axes.titlesize": 11, "axes.labelsize": 10, "font.size": 10,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ)


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div_br", [DIV_NEG, DIV_MID, DIV_POS])
