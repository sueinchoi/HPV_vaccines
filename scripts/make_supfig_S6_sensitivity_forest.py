"""Generate Supplementary Figure S6 — 5-panel forest plot summarising the
five essential pre-specified sensitivity analyses for Cohort B.

Layout: 5 vertically stacked sub-forest plots; each plot has its own x-axis
because the favourable direction differs between recurrence (HR<1) and
clearance (HR>1).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

DATA = Path(__file__).resolve().parent.parent / "Data"
OUT = DATA / "SupFigS6_Sensitivity_Forest.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


def panel(
    ax,
    text_ax,
    rows: list[tuple[str, float, float, float, str]],
    *,
    title: str,
    xlabel: str,
    favour_lt1: bool,
    xlim: tuple[float, float],
    xticks: list[float],
):
    """One mini forest plot with adjacent annotation axis.

    rows: list of (label, hr, ci_lo, ci_hi, annotation_text)
    """
    n = len(rows)
    y = np.arange(n)[::-1]

    for yi, (_label, hr, lo, hi, _ann) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color="#333", linewidth=1.4)
        marker_face = "#cc3333" if (
            (favour_lt1 and hi < 1) or (not favour_lt1 and lo > 1)
        ) else "#444"
        ax.plot(hr, yi, "s", markersize=8, markerfacecolor=marker_face,
                markeredgecolor="#222", markeredgewidth=0.6)

    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: f"{v:g}"
    ))
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold", pad=4)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(False)
    ax.set_ylim(-0.6, n - 0.4)

    # text axis on the right with annotations aligned to forest rows
    text_ax.set_xlim(0, 1)
    text_ax.set_ylim(-0.6, n - 0.4)
    for spine in text_ax.spines.values():
        spine.set_visible(False)
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    text_ax.set_title(" ", fontsize=10, pad=4)  # vertical alignment with main panel
    for yi, (_label, _hr, _lo, _hi, ann) in zip(y, rows):
        text_ax.text(0.0, yi, ann, ha="left", va="center",
                     fontsize=8, family="monospace")


def main() -> None:
    # Sens-A — single-negative vs two-consecutive clearance (v3)
    sa = read_csv(DATA / "Sensitivity_HPV_Clearance_SingleNegative_v3.csv")
    sa_rows = []
    for r in sa:
        label = r["definition"]
        hr, lo, hi = f(r["HR"]), f(r["CIlo"]), f(r["CIhi"])
        ann = (f"HR {hr:.2f} ({lo:.2f}–{hi:.2f}); "
               f"{r['ev_v']}/{r['n_v']} vs {r['ev_c']}/{r['n_c']}; p={float(r['p']):.2f}")
        sa_rows.append((label, hr, lo, hi, ann))

    # Sens-B — time-stratified clearance (v3)
    sb = read_csv(DATA / "Sensitivity_HPV_Clearance_TimeStratified_v3.csv")
    sb_rows = []
    for r in sb:
        label = r["period"]
        hr, lo, hi = f(r["HR"]), f(r["CIlo"]), f(r["CIhi"])
        ann = (f"HR {hr:.2f} ({lo:.2f}–{hi:.2f}); "
               f"{r['ev_v']}/{r['n_v']} vs {r['ev_c']}/{r['n_c']}; p={float(r['p']):.3f}")
        sb_rows.append((label, hr, lo, hi, ann))

    # Sens-C — dose threshold (Cohort B lesion recurrence)
    sc = read_csv(DATA / "Sensitivity_DoseThreshold_HR.csv")
    sc_rows = []
    for r in sc:
        if r["cohort"] == "B" and r["outcome"] == "Lesion recurrence":
            label = r["definition"]
            hr, lo, hi = f(r["HR"]), f(r["CIlo"]), f(r["CIhi"])
            ann = (f"HR {hr:.2f} ({lo:.2f}–{hi:.2f}); "
                   f"{r['ev_v']}/{r['n_v']} vs {r['ev_c']}/{r['n_c']}; p={float(r['p']):.2f}")
            sc_rows.append((label, hr, lo, hi, ann))

    # Sens-D — strict matching (lesion recurrence)
    sd = read_csv(DATA / "Sensitivity_StrictMatching.csv")
    sd_rows = []
    for r in sd:
        if r["outcome"] == "Lesion recurrence":
            label = r["design"]
            hr, lo, hi = f(r["HR"]), f(r["CIlo"]), f(r["CIhi"])
            ann = (f"HR {hr:.2f} ({lo:.2f}–{hi:.2f}); "
                   f"{r['ev_v']}/{r['n_v']} vs {r['ev_c']}/{r['n_c']}; p={float(r['p']):.2f}")
            sd_rows.append((label, hr, lo, hi, ann))

    # Sens-E — disease-free interval (lesion recurrence) (v3)
    se = read_csv(DATA / "Sensitivity_Recurrence_DFInterval_v3.csv")
    se_rows = []
    for r in se:
        label = r["definition"]
        hr, lo, hi = f(r["HR"]), f(r["CIlo"]), f(r["CIhi"])
        ann = (f"HR {hr:.2f} ({lo:.2f}–{hi:.2f}); "
               f"{r['ev_v']}/{r['n_v']} vs {r['ev_c']}/{r['n_c']}; p={float(r['p']):.2f}")
        se_rows.append((label, hr, lo, hi, ann))

    # Layout: 5 panels stacked, sized by row count.
    # Panel descriptions live in the figure legend (Sup Fig S6).
    panel_specs = [
        ("Sens-A", sa_rows,
         "Hazard ratio (HR > 1 favours vaccinated)", False,
         (0.5, 4.0), [0.5, 1.0, 2.0, 4.0]),
        ("Sens-B", sb_rows,
         "Hazard ratio (HR > 1 favours vaccinated)", False,
         (0.15, 30.0), [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]),
        ("Sens-C", sc_rows,
         "Hazard ratio (HR < 1 favours vaccinated)", True,
         (0.2, 2.5), [0.25, 0.5, 1.0, 2.0]),
        ("Sens-D", sd_rows,
         "Hazard ratio (HR < 1 favours vaccinated)", True,
         (0.25, 2.0), [0.25, 0.5, 1.0, 2.0]),
        ("Sens-E", se_rows,
         "Hazard ratio (HR < 1 favours vaccinated)", True,
         (0.25, 2.5), [0.25, 0.5, 1.0, 2.0]),
    ]

    heights = [max(1.1, 0.45 * len(rows) + 0.7) for (_, rows, *_) in panel_specs]
    total_h = sum(heights) + 0.6

    fig = plt.figure(figsize=(13.0, total_h))
    gs = fig.add_gridspec(
        nrows=len(panel_specs), ncols=2,
        height_ratios=heights, width_ratios=[1.0, 0.85],
        hspace=0.85, wspace=0.05,
    )

    for i, (title, rows, xlabel, fav_lt1, xlim, xticks) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[i, 0])
        text_ax = fig.add_subplot(gs[i, 1])
        panel(ax, text_ax, rows, title=title, xlabel=xlabel,
              favour_lt1=fav_lt1, xlim=xlim, xticks=xticks)

    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
