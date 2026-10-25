"""Supplementary figures from the immutable short-storm analysis tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
AUDIT = ROOT / "audit"
sys.path.insert(0, str(MAIN / "code"))
from audit_panel_alignment import require_matplotlib_panel_alignment

COLORS = {"graphcast": "#0072B2", "pangu": "#D55E00", "fourcastnet": "#009E73"}
LABELS = {"graphcast": "GraphCast", "pangu": "Pangu-Weather", "fourcastnet": "FourCastNet"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, name, source):
    FIGURES.mkdir(exist_ok=True, parents=True)
    AUDIT.mkdir(exist_ok=True, parents=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=AUDIT / f"{name}.alignment.json",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.svg")
    fig.savefig(FIGURES / f"{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {"figure": name, "source_table": source, "image_generation": False}


def mean_figure():
    d = pd.read_csv(TABLES / "cohort_DPE_CI_common6_36.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.83, wspace=.18)
    for ax, cohort, heading in zip(axes, ("full14_all_cases", "short13"),
                                   ("Complete storms (n=14)", "Truncated storms (n=13)")):
        for model in COLORS:
            g = d[(d.cohort == cohort) & (d.model == model)].sort_values("lead_hours")
            ax.plot(g.lead_hours, g["mean"], color=COLORS[model], marker="o", ms=2.5,
                    lw=1.3, label=LABELS[model])
            ax.fill_between(g.lead_hours, g.low, g.high, color=COLORS[model], alpha=.11, lw=0)
        ax.set_title(heading, loc="left", pad=8)
        ax.set_xticks([6, 12, 18, 24, 30, 36])
        ax.set_xlim(6, 36)
        ax.set_ylim(0, 430)
        ax.set_xlabel("Forecast lead time (h)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Mean DPE (km)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="upper center", frameon=False,
               bbox_to_anchor=(.5,.99))
    return save(fig, "Figure_S13_1_common_DPE", "cohort_DPE_CI_common6_36.csv")


def storm_figure():
    d = pd.read_csv(TABLES / "short_pair_by_storm.csv", dtype={"TC_ID": str})
    ids = sorted(d.TC_ID.unique())
    y = list(range(len(ids)))
    turns = {"2501", "2515"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.89, wspace=.20)
    for ax, pair, title in zip(axes, ("graphcast-pangu", "graphcast-fourcastnet"),
                               ("GraphCast - Pangu-Weather", "GraphCast - FourCastNet")):
        g = d[(d.pair == pair) & (d.lead_hours == 36)].set_index("TC_ID").loc[ids]
        ax.axvline(0, color=".25", lw=.8)
        for row, (tc, delta) in enumerate(g.delta.items()):
            ax.plot([0, delta], [row, row], color=".78", lw=.8)
            ax.scatter(delta, row, color="#D55E00" if tc in turns else "#0072B2", s=20, zorder=3)
        ax.axvline(g.delta.mean(), color="#D55E00", ls="--", lw=1.0)
        ax.set_xlabel("Paired DPE difference at 36 h (km)")
        ax.set_title(title, loc="left", pad=8)
        ax.set_yticks(y, [tc + ("*" if tc in turns else "") for tc in ids])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.13)
    axes[0].set_ylabel("Cyclone ID (* = diagnosed recurving)")
    return save(fig, "Figure_S13_2_individual_36h", "short_pair_by_storm.csv")


def main():
    records = [mean_figure(), storm_figure()]
    (AUDIT / "figure_manifest.json").write_text(json.dumps(records, indent=2))
    print("Created", len(records), "supplementary figures")


if __name__ == "__main__":
    main()
