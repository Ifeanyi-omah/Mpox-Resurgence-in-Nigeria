"""Supplemental figure: lineage distribution of the new Nigerian genomes in context.

Standalone. Usage:
  python plot_supp_lineages.py --meta mpox_metadata_2026-09-14T1557.tsv \
      --new Nigeria_new_genomes.csv --out Figure_S_lineage_context

Writes <out>.png/.pdf plus the four S_*.csv source tables.
See Figure_S_lineage_context_NOTES.md for date handling and caveats.
"""
import argparse, re
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--meta", default="mpox_metadata_2026-09-14T1557.tsv")
p.add_argument("--new", default="Nigeria_new_genomes.csv")
p.add_argument("--out", default="Figure_S_lineage_context")
p.add_argument("--dpi", type=int, default=400)
args = p.parse_args()

LIN_ORDER = ["A", "A.1", "A.2", "A.2.2", "A.2.3", "A.2.4", "A.2.5", "A.3", "B.1",
             "unassigned"]
# A.1.1 (n=2) and A.2.1 (n=5) are folded into their parent to keep the palette to
# ten hues; stated in the caption.
COLLAPSE = {"A.1.1": "A.1", "A.2.1": "A.2"}
GRPS = ["A (endemic Nigerian)", "B.1 (2022 global)", "C (post-2022)", "D (post-2022)",
        "E (post-2022)", "F (post-2022)", "G (post-2022)", "H (post-2022)",
        "J (post-2022)", "unassigned"]
WAVES3 = ["Wave 1\n2017-2020", "Wave 2\n2021-2023", "Wave 3\n2024-2025"]
A_RE = r"A(\.|$)"


def lingroup(l):
    if not isinstance(l, str):
        return "unassigned"
    if l == "A" or l.startswith("A."):
        return "A (endemic Nigerian)"
    if l.startswith("B.1"):
        return "B.1 (2022 global)"
    return l.split(".")[0] + " (post-2022)"


def waveof(y):
    if pd.isna(y) or y < 2017 or y > 2025:
        return None
    return WAVES3[0] if y <= 2020 else (WAVES3[1] if y <= 2023 else WAVES3[2])


def year_hint(name):
    m = re.search(r"MPO?X[-_](\d{2})[-_]", str(name).upper())
    return 2000 + int(m.group(1)) if m else np.nan


# ---- public context table: year/month by pattern (5,134 dates are partial)
M = pd.read_csv(args.meta, sep="\t", low_memory=False)
s_ = M.sampleCollectionDate.astype(str)
M["year"] = pd.to_numeric(s_.str.extract(r"^(\d{4})")[0], errors="coerce")
M["month"] = s_.str.extract(r"^(\d{4}-\d{2})")[0]
M["grp"] = M.lineage.map(lingroup)
M["sublin"] = M.lineage.fillna("unassigned").replace(COLLAPSE)
NG = M[M.geoLocCountry.eq("Nigeria")].copy()
NG["wave"] = NG.year.map(waveof)

# ---- this study: month/state parsed from the pipe-delimited seqName, year falls
#      back to the sample-ID token (same rule as Fig. 1)
NEW = pd.read_csv(args.new)
dt = NEW.seqName.astype(str).str.extract(r"\|(\d{3,4})-(\d{2})-(\d{2})\s*$")
yr = pd.to_numeric(dt[0], errors="coerce")
mo = pd.to_numeric(dt[1], errors="coerce")
NEWP = pd.DataFrame({"seqName": NEW.seqName, "lineage": NEW.lineage,
                     "year": yr.where(yr.between(2017, 2025)), "month_num": mo})
hint = NEW.seqName.map(year_hint)
NEWP["year"] = NEWP.year.fillna(hint)
NEWP["wave"] = NEWP.year.map(waveof)
# malformed year with a valid month/day (e.g. "204-10-22") is repaired from the
# sample-ID year token, as in Fig. 1
NEWP["month"] = [f"{int(y):04d}-{int(m):02d}" if pd.notna(y) and pd.notna(m) else None
                 for y, m in zip(NEWP.year, mo)]

# ---- panel tables
pubA = pd.crosstab(NG.wave, NG.lineage.fillna("unassigned").replace(COLLAPSE)).reindex(index=WAVES3, columns=LIN_ORDER).fillna(0)
newA = pd.crosstab(NEWP.wave, NEWP.lineage.fillna("unassigned").replace(COLLAPSE)).reindex(index=WAVES3, columns=LIN_ORDER).fillna(0)

inout = pd.DataFrame({
    "Nigeria\n(public + this study)": NG.grp.value_counts().reindex(GRPS).fillna(0)
        + pd.Series({"A (endemic Nigerian)": float(NEWP.lineage.notna().sum()),
                     "unassigned": float(NEWP.lineage.isna().sum())}).reindex(GRPS).fillna(0),
    "Outside Nigeria": M[~M.geoLocCountry.eq("Nigeria")].grp.value_counts().reindex(GRPS).fillna(0)})

qs = pd.period_range("2022Q1", "2026Q2", freq="Q")
Aq = M[M.lineage.astype(str).str.match(A_RE)].dropna(subset=["month"]).copy()
Aq["q"] = pd.PeriodIndex(pd.to_datetime(Aq.month + "-01"), freq="Q")
pub_q = Aq.q.value_counts().reindex(qs).fillna(0)
nq = NEWP.dropna(subset=["month"]).copy()
nq["q"] = pd.PeriodIndex(pd.to_datetime(nq.month + "-01"), freq="Q")
new_q = nq.q.value_counts().reindex(qs).fillna(0)

A_out = M[M.lineage.astype(str).str.match(A_RE) & ~M.geoLocCountry.eq("Nigeria")]
ctab = pd.crosstab(A_out.geoLocCountry, A_out.sublin)
ctab["total"] = ctab.sum(axis=1)
ctab = ctab.sort_values("total", ascending=False)

# ---- source tables
rows = [dict(wave=w.replace("\n", " "), source=src, lineage=L, genomes=int(tab.loc[w, L]))
        for src, tab in (("Public repository", pubA), ("This study", newA))
        for w in WAVES3 for L in LIN_ORDER if tab.loc[w, L]]
pd.DataFrame(rows).to_csv("S_lineage_by_wave.csv", index=False)
inout.astype(int).rename_axis("lineage_group").reset_index().to_csv(
    "S_lineage_inside_outside_nigeria.csv", index=False)
pd.DataFrame({"quarter": [str(q) for q in qs], "public_worldwide": pub_q.values.astype(int),
              "this_study_nigeria": new_q.values.astype(int)}).to_csv(
    "S_lineageA_by_quarter.csv", index=False)
ctab.rename_axis("country").reset_index().to_csv("S_lineageA_outside_nigeria.csv", index=False)

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

try:                              # session kernel: figure-style helpers present
    apply_figure_style(frame="open", font="Arial", sizes=(7, 6, 5), grid=False)
except NameError:                 # standalone: Nature-equivalent rcParams
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "legend.fontsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.minor.size": 1.2,
        "xtick.minor.width": 0.4, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
    })

    def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=8):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold", va="bottom", ha="left")

# lineage palette: family = hue (A.1 blues, A.2.x oranges, A.3 purple, B.1 teal),
# sublineage = tint within family; unresolved / unassigned in neutral greys
PAL = {"A": "#9C9C9C", "A.1": "#6BAED6", "A.2": "#FDD0A2", "A.2.2": "#E6550D",
       "A.2.3": "#8C2D04", "A.2.4": "#FDAE6B", "A.2.5": "#A6761D", "A.3": "#6A51A3",
       "B.1": "#238B8B", "unassigned": "#E0E0E0"}
# lineage-group palette (panel b) - Okabe-Ito, deliberately distinct from PAL
GPAL = {"A (endemic Nigerian)": "#D55E00", "B.1 (2022 global)": "#009E73",
        "C (post-2022)": "#56B4E9", "D (post-2022)": "#0072B2", "E (post-2022)": "#CC79A7",
        "F (post-2022)": "#5D3A9B", "G (post-2022)": "#E8C300", "H (post-2022)": "#4D4D4D",
        "J (post-2022)": "#BFBFBF", "unassigned": "#E8E8E8"}
GREY, FOCAL = "#BEBEBE", "#D55E00"

# 183 mm double-column width
fig = plt.figure(figsize=(7.2, 5.5))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                      left=0.075, right=0.985, top=0.93, bottom=0.095)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

# ------------------------------------------------------------------ a  waves
xs, w = np.arange(3), 0.30
for off, tab, lab in ((-0.17, pubA, "Public"), (0.17, newA, "This study")):
    tot = tab.sum(axis=1).values
    bot = np.zeros(3)
    for L in LIN_ORDER:
        pct = np.where(tot > 0, 100 * tab[L].values / np.where(tot > 0, tot, 1), 0)
        axA.bar(xs + off, pct, width=w, bottom=bot, color=PAL[L],
                edgecolor="white", linewidth=0.45, zorder=3)
        bot += pct
    for i, n in enumerate(tot):
        axA.text(xs[i] + off, 101.0 if n else 1.0, ("n = %d" % n) if n else "none",
                 ha="center", va="bottom", fontsize=5,
                 color="black" if n else "#8A8A8A")
        axA.text(xs[i] + off, -3.0, lab, ha="center", va="top", fontsize=5, color="#3A3A3A")
axA.set_xticks(xs)
axA.set_xticklabels(["Wave 1\n2017\u20132020", "Wave 2\n2021\u20132023", "Wave 3\n2024\u20132025"],
                    fontsize=6)
axA.tick_params(axis="x", pad=11, length=0)
axA.set_xlim(-0.55, 2.55)
axA.set_ylim(0, 119)
axA.set_yticks([0, 25, 50, 75, 100])
axA.set_ylabel("Genomes (%)")
axA.legend(handles=[Patch(facecolor=PAL[L], edgecolor="white", linewidth=0.3, label=L)
                    for L in LIN_ORDER],
           ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.035), frameon=False,
           fontsize=5, handlelength=0.9, handleheight=0.9, handletextpad=0.4,
           labelspacing=0.28, columnspacing=0.8, borderpad=0.0)
panel_letter(axA, "a", dx=-0.115, dy=1.02)

# ------------------------------------------------- b  inside vs outside Nigeria
cols = list(inout.columns)
ys = np.array([1.0, 0.0])
tot_io = inout.sum(axis=0).values
left = np.zeros(2)
for G in GRPS:
    pct = 100 * inout.loc[G].values / tot_io
    axB.barh(ys, pct, height=0.46, left=left, color=GPAL[G],
             edgecolor="white", linewidth=0.45, zorder=3)
    for i, (p, l) in enumerate(zip(pct, left)):
        if p >= 9:
            axB.text(l + p / 2, ys[i], G.split(" ")[0], ha="center", va="center",
                     fontsize=6, color="white", zorder=4)
    left += pct
for i, (c, n) in enumerate(zip(cols, tot_io)):
    axB.text(-1.5, ys[i] + 0.06, c.split("\n")[0].split(" (")[0], ha="right",
             va="bottom", fontsize=6)
    axB.text(-1.5, ys[i] - 0.06, "n = %s" % format(int(n), ","), ha="right",
             va="top", fontsize=5, color="#3A3A3A")
axB.set_xlim(0, 100)
axB.set_ylim(-0.55, 1.90)
axB.set_yticks([])
axB.set_xlabel("Genomes (%)")
axB.spines["left"].set_visible(False)
axB.tick_params(axis="y", length=0)
axB.legend(handles=[Patch(facecolor=GPAL[G], edgecolor="white", linewidth=0.3, label=G)
                    for G in GRPS if inout.loc[G].sum() > 0],
           ncol=3, loc="upper center", bbox_to_anchor=(0.46, 1.10), frameon=False,
           fontsize=5, handlelength=0.9, handleheight=0.9, handletextpad=0.4,
           labelspacing=0.28, columnspacing=0.9, borderpad=0.0)
panel_letter(axB, "b", dx=-0.115, dy=1.02)

# --------------------------------------------- c  lineage A sampling over time
xq = np.arange(len(qs))
axC.bar(xq, pub_q.values, width=0.74, color=GREY, edgecolor="white", linewidth=0.25,
        zorder=3, label="Public repository, all countries")
axC.bar(xq, new_q.values, width=0.74, bottom=pub_q.values, color=FOCAL,
        edgecolor="white", linewidth=0.25, zorder=3, label="This study, Nigeria")
yr_ticks = [i for i, q in enumerate(qs) if q.quarter == 1]
axC.set_xticks(yr_ticks)
axC.set_xticklabels([str(qs[i].year) for i in yr_ticks], fontsize=6)
axC.set_xticks(xq, minor=True)
axC.set_xlim(-0.8, len(qs) - 0.2)
axC.set_ylabel("Lineage A genomes")
axC.set_xlabel("Collection quarter")
axC.yaxis.set_major_locator(MultipleLocator(20))
axC.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=5.5,
           handlelength=0.9, handleheight=0.9, handletextpad=0.4, labelspacing=0.3,
           borderpad=0.0)
panel_letter(axC, "c", dx=-0.115, dy=1.02)

# ------------------------------------------ d  lineage A sampled outside Nigeria
sub = ctab.drop(columns="total")
yy = np.arange(len(sub))[::-1]
left = np.zeros(len(sub))
for L in LIN_ORDER:
    if L in sub.columns:
        v = sub[L].values.astype(float)
        axD.barh(yy, v, height=0.62, left=left, color=PAL[L], edgecolor="white",
                 linewidth=0.3, zorder=3)
        left += v
axD.set_yticks(yy)
axD.set_yticklabels(sub.index, fontsize=5.5)
axD.set_ylim(-0.7, len(sub) - 0.3)
axD.set_xlim(0, np.ceil(left.max() / 5) * 5)
axD.xaxis.set_major_locator(MultipleLocator(5))
axD.set_xlabel("Genomes")
axD.tick_params(axis="y", length=0, pad=1.5)
panel_letter(axD, "d", dx=-0.185, dy=1.02)

fig.savefig(args.out + ".png", dpi=600, facecolor="white")
fig.savefig(args.out + ".pdf", facecolor="white")
print("wrote %s.png/.pdf (183 mm) and 4 S_*.csv tables" % args.out)
