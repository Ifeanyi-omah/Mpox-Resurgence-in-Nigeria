#!/usr/bin/env python3
"""
mpox_extended_state_tree.py
===============================================================================
EXTENDED DATA FIGURE — Nigerian mpox exploded state DTA tree (standalone).

This is the former panel A of the composite, rendered on its own at
Extended-Data scale: one full page, larger tips, larger state labels,
region legend, and (optionally) weekly case bars behind each state block.

Inputs (all expected in ~/Downloads):
    NCDC_2026_state_HPSTR.tree        state DTA MCC/HPSTR tree ('location' trait)
    1Mpox_NCDC_cases_correct.csv      OPTIONAL weekly case line-list; skipped if absent

Outputs:
    Extended_Data_state_DTA_tree.{png,pdf,svg}

pip install numpy pandas matplotlib baltic
===============================================================================
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime as py_datetime

import numpy as np
import pandas as pd
import baltic as bt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


# =============================================================================
# Paths
# =============================================================================
BASE = Path("/Users/s2059633/Downloads")

STATE_TREE_PATH = BASE / "NCDC_2026_state_HPSTR.tree"
STATE_TRAIT = "location"

# Optional: weekly case bars behind each state block. Set to None to disable.
CASES_CSV = BASE / "1Mpox_NCDC_cases_correct.csv"

OUT_BASE = BASE / "Extended_Data_state_DTA_tree"


# =============================================================================
# Key settings
# =============================================================================
TREE_MOST_RECENT = 2025.786301369863          # 2025-10-15
TIP_REGEX = r"([0-9]{4}-[0-9]{2}-[0-9]{2})$"  # tips end ...|YYYY-MM-DD
DATE_FMT = "%Y-%m-%d"

XLIM = (2015.6, 2026.2)
SEP_2017 = True                                # red dashed line at 2017-09-22
SHOW = False                                   # plt.show() at the end


# =============================================================================
# Figure style
# =============================================================================
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 1.6
plt.rcParams["font.size"] = 22
plt.rcParams["text.usetex"] = False

FIG_W, FIG_H = 15.0, 21.0     # inches; full Extended-Data page

FS_TICK = 26
FS_LEGEND = 26
TREE_LABEL_FS = 26
branchWidth = 3.4
TIP_SIZE = 130
ORIGIN_SIZE = 520
MIN_STATE_BLOCK = 20          # min vertical units per state block (no label collisions)


# =============================================================================
# Palettes and state/region utilities
# =============================================================================
REGION_COLORS = {
    "SS": "#D27D2D", "SW": "#77BEDB", "SE": "#48A365",
    "NC": "#7F6E85", "NE": "#E1C72F", "NW": "#D1684A",
    "Rest": "#C8C8C8", "Unknown": "#C8C8C8",
}
REGION_LABELS = {
    "SS": "South South", "SW": "South West", "SE": "South East",
    "NC": "North Central", "NE": "North East", "NW": "North West",
}
REGION_ORDER = ["SW", "SS", "SE", "NC", "NW", "NE"]
REGION_RANK = {r: i for i, r in enumerate(REGION_ORDER)}

STATE_TO_REGION = {
    "Abia": "SE", "Anambra": "SE", "Ebonyi": "SE", "Enugu": "SE", "Imo": "SE",
    "AkwaIbom": "SS", "Bayelsa": "SS", "CrossRiver": "SS", "Delta": "SS", "Edo": "SS", "Rivers": "SS",
    "Ekiti": "SW", "Lagos": "SW", "Ogun": "SW", "Ondo": "SW", "Osun": "SW", "Oyo": "SW",
    "Benue": "NC", "FCT": "NC", "Kogi": "NC", "Kwara": "NC", "Nasarawa": "NC", "Niger": "NC", "Plateau": "NC",
    "Adamawa": "NE", "Bauchi": "NE", "Borno": "NE", "Gombe": "NE", "Taraba": "NE", "Yobe": "NE",
    "Jigawa": "NW", "Kaduna": "NW", "Kano": "NW", "Katsina": "NW", "Kebbi": "NW", "Sokoto": "NW", "Zamfara": "NW",
}

STATE_NAME_FIX = {
    "Akwa Ibom": "AkwaIbom", "Akwa_Ibom": "AkwaIbom", "AkwaIbom": "AkwaIbom",
    "Cross River": "CrossRiver", "Cross_River": "CrossRiver", "CrossRiver": "CrossRiver",
    "Federal Capital Territory": "FCT", "Federal_Capital_Territory": "FCT",
    "FederalCapitalTerritory": "FCT", "Abuja Federal Capital Territory": "FCT",
    "FCT Abuja": "FCT", "Abuja": "FCT",
    "Nassarawa": "Nasarawa",
    "South South": "SS", "South_South": "SS", "South West": "SW", "South_West": "SW",
    "South East": "SE", "South_East": "SE", "North Central": "NC", "North_Central": "NC",
    "North East": "NE", "North_East": "NE", "North West": "NW", "North_West": "NW",
}


def clean_location(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = str(x).strip().strip('"').strip("'")
    x = " ".join(x.replace("_", " ").replace("-", " ").split())
    if x in STATE_NAME_FIX:
        return STATE_NAME_FIX[x]
    for k, v in STATE_NAME_FIX.items():
        if x.lower() == k.lower():
            return v
    if x in REGION_LABELS or x in REGION_COLORS:
        return x
    return x.replace(" ", "")


def pretty_location_label(x):
    x = clean_location(x)
    if x == "FCT":
        return "FCT"
    return REGION_LABELS.get(x, x)


def region_of(value):
    value = clean_location(value)
    if value in STATE_TO_REGION:
        return STATE_TO_REGION[value]
    if value in REGION_RANK:
        return value
    return "Unknown"


colors = {st: REGION_COLORS[reg] for st, reg in STATE_TO_REGION.items()}
for code, col in REGION_COLORS.items():
    colors[code] = col
for code, full in REGION_LABELS.items():
    colors[full.replace(" ", "")] = REGION_COLORS[code]
colors["ancestor"] = "#B9B9B9"


# =============================================================================
# Utilities
# =============================================================================
def convertDate(x, start, end):
    return py_datetime.strftime(py_datetime.strptime(x, start), end)


def save_multi(fig, out_base, dpi=300):
    base = Path(out_base).with_suffix("")
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(f"{base}{ext}", bbox_inches="tight", dpi=dpi)
    print(f"wrote {base}.png / .pdf / .svg")


def load_tree(path):
    """Load the DTA tree, tolerating baltic signature differences."""
    try:
        tree = bt.loadNexus(str(path), tip_regex=TIP_REGEX, date_fmt=DATE_FMT)
    except TypeError:
        tree = bt.loadNexus(str(path), tip_regex=TIP_REGEX)
    tree.setAbsoluteTime(TREE_MOST_RECENT)
    return tree


# =============================================================================
# Exploded tree
# =============================================================================
def enumerate_subtrees(tree, trait="location"):
    tree.root.traits[trait] = "ancestor"
    subtype_trees = defaultdict(list)
    for k in sorted(tree.Objects, key=lambda x: x.height):
        kp = k.parent
        kloc = clean_location(k.traits[trait])
        k.traits[trait] = kloc
        kploc = clean_location(kp.traits[trait]) if (kp is not None and trait in kp.traits) else "ancestor"
        if kloc != kploc:
            cond = lambda w, _kloc=kloc: clean_location(w.traits[trait]) == _kloc
            subtree = tree.subtree(k, traverse_condition=cond)
            if subtree is not None:
                subtree.traverse_tree()
                subtree.sortBranches()
                subtype_trees[kloc].append((kploc, subtree))
    return dict(subtype_trees)


def load_weekly_cases(path):
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    cols = [c for c in ["location", "date", "cases"] if c in raw.columns]
    raw = raw[cols].copy()
    raw["location"] = raw["location"].apply(clean_location)
    raw["cases"] = pd.to_numeric(raw["cases"], errors="coerce").fillna(0)
    weekly = (
        raw.groupby(["location", pd.Grouper(key="date", freq="W-Wed", label="right", closed="right")], as_index=False)
        .agg({"cases": "sum"})
        .sort_values(["location", "date"])
    )
    weekly["d_date"] = weekly["date"].astype(str).apply(lambda x: bt.decimalDate(x))
    return weekly


def draw_exploded_tree(ax1, subtype_trees, *, trait_name="location", weekly_data=None,
                       xlim=XLIM, legend=True, sep_2017=SEP_2017):
    present = list(subtype_trees.keys())
    heights = {d: [] for d in present}
    cumulative_y = 0

    def state_sort_key(s):
        return (REGION_RANK.get(region_of(s), 999), s)

    for state in reversed(sorted(subtype_trees.keys(), key=state_sort_key)):
        starting_y = cumulative_y
        cumulative_y += 8
        for origin_state, loc_tree in subtype_trees[state]:
            origin_state = clean_location(origin_state)
            for k in loc_tree.Objects:
                k_state = clean_location(k.traits.get(trait_name))
                x = k.absoluteTime
                y = k.y + cumulative_y
                c = colors.get(k_state, "#C0C0C0")
                a = k.traits.get(f"{trait_name}.prob", 1.0)
                if k.branchType == "leaf":
                    ax1.scatter(x, y, s=TIP_SIZE, facecolor=c, alpha=a, edgecolor="black",
                                linewidth=0.4, zorder=100)
                elif k.branchType == "node":
                    ax1.plot([x, x], [k.children[-1].y + cumulative_y, k.children[0].y + cumulative_y],
                             lw=branchWidth, color=c, alpha=a, ls="-", zorder=9)
                kp = getattr(k, "parent", None)
                if kp is not None and getattr(kp, "absoluteTime", None) is not None:
                    ax1.plot([kp.absoluteTime, x], [y, y], lw=branchWidth, color=c, ls="-", zorder=9)

            oriC = colors.get(origin_state, "#B9B9B9")
            root_parent = getattr(loc_tree.root, "parent", None)
            if root_parent is not None and getattr(root_parent, "absoluteTime", None) is not None:
                oriX, oriY = root_parent.absoluteTime, loc_tree.root.y + cumulative_y
                if root_parent.traits.get("posterior", 1.0) >= 0.5:
                    ax1.scatter(oriX, oriY, s=ORIGIN_SIZE, facecolor=oriC, edgecolor="white",
                                lw=1.4, zorder=200)
                else:
                    ax1.scatter(oriX, oriY, s=ORIGIN_SIZE * 0.5, facecolor=oriC, edgecolor="black",
                                lw=0.9, zorder=200)
            else:
                ax1.scatter(loc_tree.root.absoluteTime, loc_tree.root.y + cumulative_y,
                            s=ORIGIN_SIZE * 0.5, facecolor=oriC, edgecolor="black", lw=0.9, zorder=200)
            cumulative_y += loc_tree.ySpan + 6

        if (cumulative_y - starting_y) < MIN_STATE_BLOCK:
            cumulative_y = starting_y + MIN_STATE_BLOCK
        ax1.axhspan(cumulative_y, starting_y, facecolor=colors.get(state, "#E0E0E0"), alpha=0.12)
        heights[state].append((starting_y, cumulative_y))

    if weekly_data is not None:
        for state in reversed(sorted(subtype_trees.keys(), key=state_sort_key)):
            start, end = heights[state][0]
            df_state = weekly_data.loc[weekly_data["location"] == state]
            if df_state.empty:
                continue
            x = df_state["d_date"].values
            cases = df_state["cases"].values
            denom = cases.max() - cases.min()
            scale = (end - start) * (0.75 / denom) if denom != 0 else 0.0
            ax1.bar(x, (cases - cases.min()) * scale, color=colors.get(state, "#C0C0C0"),
                    width=0.01, alpha=0.18, bottom=start, linewidth=0)

    if legend:
        present_regions = sorted({region_of(s) for s in present if region_of(s) != "Unknown"},
                                 key=lambda r: REGION_RANK.get(r, 999))
        handles = [mlines.Line2D([], [], linestyle="None", marker="o", markersize=16,
                                 markerfacecolor=REGION_COLORS[r], markeredgecolor="black",
                                 label=REGION_LABELS[r]) for r in present_regions]
        ax1.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0),
                   prop={"size": FS_LEGEND}, frameon=False, ncol=1)

    tick_years = range(2016, 2027, 1)
    tick_dates = [f"{yr:04d}-01-01" for yr in tick_years]
    ax1.set_xticks([bt.decimalDate(d) for d in tick_dates])
    ax1.set_xticklabels([convertDate(d, "%Y-%m-%d", "%Y") for d in tick_dates])
    ax1.tick_params(axis="x", labelsize=FS_TICK, size=7)

    ax1.yaxis.tick_left()
    for loc in ("top", "right", "left"):
        ax1.spines[loc].set_visible(False)
    ax1.tick_params(axis="y", size=0)
    ax1.set_yticklabels([])
    ax1.set_ylim(-5, cumulative_y)
    ax1.set_xlim(*xlim)

    boundaries = set()
    for state in subtype_trees:
        s0, e0 = heights[state][0]
        boundaries.update((s0, e0))
    for yb in sorted(boundaries):
        ax1.axhline(y=yb, xmin=0, xmax=1, color="#666666", lw=1.0, ls="--", alpha=0.7, zorder=150)

    trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData)
    for state in reversed(sorted(subtype_trees.keys(), key=state_sort_key)):
        s0, e0 = heights[state][0]
        ax1.text(0.015, 0.5 * (s0 + e0), pretty_location_label(state), transform=trans,
                 ha="right", va="center", fontsize=TREE_LABEL_FS,
                 color=colors.get(state, "#333333"), clip_on=False, zorder=500)

    if sep_2017:
        ax1.axvline(x=bt.decimalDate("2017-09-22"), color="red", lw=2.0, ls="--", zorder=300)


# =============================================================================
# Main
# =============================================================================
def main():
    if not STATE_TREE_PATH.exists():
        raise FileNotFoundError(f"Tree not found: {STATE_TREE_PATH}")

    print("Loading exploded DTA tree...")
    state_tree = load_tree(STATE_TREE_PATH)
    state_subtrees = enumerate_subtrees(state_tree, trait=STATE_TRAIT)
    print(f"  {len(state_subtrees)} states with at least one introduction")

    weekly = None
    if CASES_CSV is not None and Path(CASES_CSV).exists():
        print("Loading weekly case counts...")
        weekly = load_weekly_cases(CASES_CSV)
    else:
        print("No case CSV found; drawing tree without case bars.")

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")
    ax = fig.add_subplot(111)
    draw_exploded_tree(ax, state_subtrees, trait_name=STATE_TRAIT,
                       weekly_data=weekly, legend=True)
    ax.set_xlabel("Year", fontsize=FS_TICK + 2, labelpad=10)

    save_multi(fig, OUT_BASE, dpi=300)
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
