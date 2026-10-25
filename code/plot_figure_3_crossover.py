"""Render the GraphCast-Pangu Weather DPE and RMSE crossover figure."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path("/Users/fei/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    }
)


def paired_values(source: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = source.pivot(index="lead_hours", columns="model", values=metric)
    result = (pivot["graphcast"] - pivot["pangu"]).rename("difference").reset_index()
    return result.sort_values("lead_hours")


def plot_panel(ax: plt.Axes, data: pd.DataFrame, color: str, title: str, crossover: str, label: str) -> None:
    ax.axhline(0, color="#43535d", linewidth=1.4, zorder=1)
    ax.plot(
        data["lead_hours"], data["difference"], color=color, linewidth=2.8,
        marker="o", markersize=4.8, markeredgewidth=0, zorder=3,
    )
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=14)
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=17, fontweight="bold")
    ax.text(
        0.98, 0.96, f"交叉区间：{crossover}", transform=ax.transAxes,
        ha="right", va="top", color=color, fontsize=11.5, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )
    ax.set_xlim(6, 120)
    ax.set_ylim(-170, 170)
    ax.set_xticks([6, 24, 48, 72, 96, 120])
    ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])
    ax.grid(axis="both", color="#dbe4e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("预报时效 / h", fontsize=12, labelpad=9)
    ax.set_ylabel("GraphCast − Pangu-Weather / km", fontsize=12, labelpad=10)
    ax.tick_params(labelsize=11, pad=5)


def main() -> None:
    dpe = pd.read_csv(ROOT / "data" / "overall_by_lead.csv")
    dpe = dpe.loc[dpe["analysis_set"].eq("primary_full120")]
    rmse = pd.read_csv(ROOT / "data" / "rmse_by_lead.csv")

    dpe_pair = paired_values(dpe, "mean_error_km_storm_weighted")
    rmse_pair = paired_values(rmse, "rmse_km")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.25), dpi=300, sharey=True)
    plot_panel(axes[0], dpe_pair, "#1f6fae", "平均DPE差值", "66–72 h", "a")
    plot_panel(axes[1], rmse_pair, "#c96b32", "RMSE差值", "72–78 h", "b")
    fig.text(
        0.5, 0.018,
        "负值表示GraphCast平均误差较小，正值表示Pangu-Weather平均误差较小",
        ha="center", fontsize=11.5, color="#43535d",
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.17, top=0.88, wspace=0.16)
    fig.canvas.draw()

    output = ROOT / "figures" / "Figure_3_crossover"
    require_matplotlib_panel_alignment(
        fig,
        json_out=f"{output}.alignment.json",
        overlay_svg=f"{output}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(f"{output}.svg", bbox_inches="tight")
    fig.savefig(f"{output}.pdf", bbox_inches="tight")
    fig.savefig(f"{output}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
