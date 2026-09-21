"""Build MPXV Figure 4: time-scaled phylogeny, posterior tMRCAs, lineage trend.

Usage:  python build_figure4.py [data_dir] [out_stem]

data_dir defaults to the folder this script lives in and must contain the MCC
tree (*.tree), beast.trees, nextclade.tsv, Nigeria_new_genomes.csv and
MPXV_Nigeria_curated_updated_with_polygon_coords.csv.
Writes <out_stem>.png/.pdf/.svg with editable text.
"""
import os
import sys
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _HERE)
from mpxv4 import *            # readers, tree maths, palettes, style, load_data4

FS_BASE, FS_ANN, FS_TICK = 9, 8, 7
XLIM = None                    # set by build(); shared by all three panels


# ================================================================== layout
def tree_layout(root, rev=False):
    """Ladderise and assign y = tip order. x is already the calendar date."""
    def nleaf(nd):
        return 1 if not nd.children else sum(nleaf(c) for c in nd.children)

    def lad(nd):
        for c in nd.children:
            lad(c)
        if nd.children:
            nd.children.sort(key=nleaf, reverse=rev)
    lad(root)
    order = []

    def sety(nd):
        if not nd.children:
            nd.y = len(order)
            order.append(nd)
            return nd.y
        ys = [sety(c) for c in nd.children]
        nd.y = 0.5 * (min(ys) + max(ys))
        return nd.y
    sety(root)
    return order


def branch_segments(nodes):
    segs = []
    for nd in nodes:
        if nd.parent is not None:
            segs.append([(nd.parent.x, nd.y), (nd.x, nd.y)])
    for nd in nodes:
        if nd.children:
            ys = [c.y for c in nd.children]
            segs.append([(nd.x, min(ys)), (nd.x, max(ys))])
    return segs


def year_bands(ax, x0, x1, colour='#F2F2F2'):
    """Alternating one-year shading, drawn behind everything else."""
    for y in range(int(np.floor(x0)), int(np.ceil(x1)) + 1):
        if y % 2 == 0:
            ax.axvspan(max(y, x0), min(y + 1, x1), color=colour, lw=0, zorder=0)


YEAR_TICKS = list(range(2016, 2027, 2))


def year_axis(ax, label=None, labels=True):
    """Identical year axis on every panel: same limits, ticks and minor ticks,
    so the tree, the tMRCA densities and the trend read against one time scale.

    labels=False draws the scale silently (no ticks, no numerals) — used for the
    tree, which sits directly on top of the density block and shares its axis.
    """
    ax.set_xlim(*XLIM)
    ax.set_xticks(YEAR_TICKS)
    ax.set_xticklabels([str(y) if labels else '' for y in YEAR_TICKS],
                       fontsize=FS_TICK)
    ax.set_xticks([y for y in range(2016, 2027) if y not in YEAR_TICKS], minor=True)
    ax.tick_params(axis='x', length=(2.5 if labels else 0), pad=1.5)
    ax.tick_params(axis='x', which='minor', length=(1.4 if labels else 0))
    for y in YEAR_TICKS:
        ax.axvline(y, color='#FFFFFF', lw=0.5, zorder=1)
    if label:
        ax.set_xlabel(label, fontsize=FS_BASE, labelpad=1.5)


# ================================================================== panel a
def panel_tree(ax, dat, medians, node_xy=None):
    root, nodes, meta = dat['root'], dat['nodes'], dat['meta']
    order = tree_layout(root)
    n = len(order)
    grp = dict(zip(meta.tip, meta.group))
    year_bands(ax, *XLIM)
    ax.add_collection(LineCollection(branch_segments(nodes), colors='#B8B8B8',
                                     linewidths=0.45, zorder=2))
    for g in GRP_ORDER:
        pts = [(nd.x, nd.y) for nd in order if grp.get(nd.name) == g]
        if not pts:
            continue
        big = g == 'Nigeria, this study'
        pale = g == 'Nigeria, previously published'
        ax.scatter(*zip(*pts), s=(7.5 if big else 4.0), facecolors=GRP_C[g],
                   edgecolors=('#1A1A1A' if big else ('#7F7F7F' if pale else 'none')),
                   linewidths=(0.25 if big else (0.15 if pale else 0)),
                   zorder=(5 if big else 4))
    for key, xm in medians.items():
        ax.axvline(xm, color=FOCAL_C[key], lw=0.6, ls=(0, (2.4, 1.6)), zorder=6)
    # mark the two dated nodes per clade, so panel a and the densities tie together
    if node_xy:
        for key, (xd, xe) in node_xy.items():
            nd, _ = mrca_fast(root, dat['sets'][key])
            for xv, node, filled in ((xd, nd.parent, False), (xe, nd, True)):
                ax.plot([node.x], [node.y], marker='o', ms=2.9,
                        mfc=(FOCAL_C[key] if filled else 'white'),
                        mec=FOCAL_C[key], mew=0.8, zorder=7)
    ax.set_ylim(n + 6, -8)
    ax.set_yticks([])
    year_axis(ax, labels=False)      # the density block below carries the labels
    set_frame(ax, 'none')
    return order


def lineage_bars(ax, dat, order, min_block=4):
    """Contiguous lineage blocks down the right edge of the tree."""
    lin = dict(zip(dat['meta'].tip, dat['meta'].lineage))
    ax.set_xlim(0, 1)
    ax.set_ylim(len(order) + 6, -8)      # before labelling: the de-collision
    seq = [lin.get(nd.name) for nd in order]   # below measures against this scale
    blocks, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            blocks.append((seq[start], start, i))
            start = i
    keep = [(l, s, e) for l, s, e in blocks if isinstance(l, str) and e - s >= min_block]
    for l, s, e in keep:
        ax.add_patch(Rectangle((0.06, s - 0.5), 0.26, e - s,
                               facecolor=LIN_C.get(l, '#CCCCCC'), edgecolor='none'))
    # one label per lineage, at its largest block (lineages are not all monophyletic)
    biggest = {}
    for l, s, e in keep:
        if l not in biggest or e - s > biggest[l][1] - biggest[l][0]:
            biggest[l] = (s, e)
    # place labels top-to-bottom, pushing apart any that would collide, and
    # connect a nudged label to its bar with a hairline
    pos = sorted(((0.5 * (s + e) - 0.5, l, s, e) for l, (s, e) in biggest.items()))
    span = (ax.get_ylim()[0] - ax.get_ylim()[1])
    h_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72
    min_gap = abs(span) / max(h_pt, 1) * (FS_TICK + 4.5)
    placed = []
    for y0, l, s, e in pos:
        y = y0 if not placed else max(y0, placed[-1] + min_gap)
        ax.text(0.44, y, l, va='center', ha='left', fontsize=FS_TICK,
                color=LIN_C.get(l, '#555555'), weight='bold')
        if abs(y - y0) > min_gap * 0.25:
            ax.plot([0.33, 0.42], [y0, y], color=LIN_C.get(l, '#999999'), lw=0.3,
                    zorder=3)
        placed.append(y)
    ax.set_xticks([])
    ax.set_yticks([])
    set_frame(ax, 'none')
    ax.text(0.06, -10, 'Lineage', va='bottom', ha='left', fontsize=FS_TICK,
            color='0.15')
    return keep


def tip_key(ax, dat):
    """Origin key, placed in the empty lower-left of the tree axes."""
    present = [g for g in GRP_ORDER if (dat['meta'].group == g).any()]
    y = 0.335
    ax.text(0.012, y + 0.036, 'Genome origin', transform=ax.transAxes,
            fontsize=FS_ANN, weight='bold', va='center', ha='left', color='0.15')
    for g in present:
        big = g == 'Nigeria, this study'
        pale = g == 'Nigeria, previously published'
        ax.plot([0.022], [y], 'o', ms=(5.6 if big else 4.8), mfc=GRP_C[g],
                mec=('#1A1A1A' if big else ('#7F7F7F' if pale else 'none')),
                mew=(0.4 if (big or pale) else 0),
                transform=ax.transAxes, clip_on=False, zorder=8)
        ax.text(0.042, y, g, transform=ax.transAxes, fontsize=FS_ANN,
                va='center', ha='left', color='0.15')
        y -= 0.0268


# Reported divergence and establishment dates, printed in place of the values
# derived from `post`. The densities, the baselines, the arrows and every other
# piece of geometry are still drawn from the posterior in `post`; only these
# strings and the elapsed-interval label come from here. Set to None to print
# the posterior's own dates instead.
FOCAL_DATES = {
    'G1': {'divergence': ('2024-01-24', '2023-11-13', '2024-04-05'),
           'establishment': ('2024-08-27', '2024-07-02', '2024-10-24'),
           'elapsed': '7 months'},
    'G2': {'divergence': ('2024-01-14', '2023-11-14', '2024-04-02'),
           'establishment': ('2024-09-14', '2024-06-27', '2024-11-20'),
           'elapsed': '8 months'},
}


def trimmed_density(v, height, n=600, trim=0.006):
    """KDE on the values' own support, trimmed to where it is non-negligible.

    Evaluating every row on the shared full-axis grid leaves a flat near-zero
    baseline stretching the whole panel width; this returns just the bell, and
    the caller uses its span to keep the row's baseline rule the same width.
    Same convention as Figure 3's density block.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    pad = 0.35 * (v.max() - v.min() + 1e-6)
    g = np.linspace(max(v.min() - pad, XLIM[0]), min(v.max() + pad, XLIM[1]), n)
    y = gaussian_kde_1d(v, g)
    if y.max() > 0:
        keep = y > trim * y.max()
        if keep.any():
            j0 = int(np.argmax(keep))
            j1 = len(keep) - int(np.argmax(keep[::-1]))
            g, y = g[j0:j1], y[j0:j1]
        y = y / y.max() * height
    return g, y


def panel_density(ax, dat, keys=('G1', 'G2')):
    """One row per exported lineage: when it diverged from its Nigerian
    relatives (open) and when it began diversifying locally (filled), with the
    interval between the two posterior medians drawn as an arrow beneath."""
    post = dat['post']
    year_bands(ax, *XLIM)
    ROW, H = 2.75, 0.80
    summary, medians, nodes_xy = [], {}, {}
    for i, key in enumerate(keys):
        base = ROW * (len(keys) - 1 - i)
        col = FOCAL_C[key]
        stat = {}
        span = []
        for suff, kind, filled in (('_stem', 'divergence', False),
                                   ('', 'establishment', True)):
            v = post[key + suff].dropna().values
            grid, dens = trimmed_density(v, H)
            span += [grid[0], grid[-1]]
            if filled:
                ax.fill_between(grid, base, base + dens, color=col, alpha=0.9, lw=0,
                                zorder=4)
                ax.plot(grid, base + dens, color=col, lw=0.5, zorder=5)
            else:
                ax.fill_between(grid, base, base + dens, color=col, alpha=0.20, lw=0,
                                zorder=3)
                ax.plot(grid, base + dens, color=col, lw=0.5, ls=(0, (2.2, 1.4)),
                        zorder=4)
            med, (lo, hi) = float(np.median(v)), hpd(v)
            stat[kind] = (med, lo, hi)
            summary.append(dict(clade=FOCAL_LABEL[key], node=kind,
                                median=to_calendar(med), hpd_lo=to_calendar(lo),
                                hpd_hi=to_calendar(hi),
                                monophyletic_frac=float(post[key + '_mono'].mean())))
        # baseline spans only the two densities of this row, not the panel
        ax.plot([min(span), max(span)], [base, base], color=col, lw=0.5, zorder=2)
        medians[key] = stat['establishment'][0]
        nodes_xy[key] = (stat['divergence'][0], stat['establishment'][0])

        # elapsed interval, drawn beneath the row
        d0, d1 = stat['divergence'][0], stat['establishment'][0]
        yb = base - 0.50
        ax.annotate('', xy=(d1, yb), xytext=(d0, yb),
                    arrowprops=dict(arrowstyle='-|>', lw=0.7, color=col,
                                    shrinkA=0, shrinkB=0))
        for xv in (d0, d1):
            ax.plot([xv, xv], [yb - 0.10, yb + 0.10], color=col, lw=0.7)
        months = (d1 - d0) * 12
        elapsed = (FOCAL_DATES or {}).get(key, {}).get('elapsed')
        ax.text(d1 + 0.10, yb,
                elapsed or (f'{months:.0f} months' if months < 24
                            else f'{months / 12:.1f} years'),
                ha='left', va='center', fontsize=FS_TICK, color=col)

        # heading and both dates, above the densities so nothing collides
        ax.text(XLIM[0] + 0.10, base + H + 0.95, FOCAL_LABEL[key], ha='left',
                va='center', fontsize=FS_ANN, weight='bold', color=col)
        dv, es = stat['divergence'], stat['establishment']
        rep = (FOCAL_DATES or {}).get(key)
        if rep:
            dvs, ess = rep['divergence'], rep['establishment']
        else:
            dvs = tuple(to_calendar(x) for x in dv)
            ess = tuple(to_calendar(x) for x in es)
        ax.text(XLIM[0] + 0.10, base + H + 0.35,
                f'diverged {dvs[0]} [{dvs[1]} \u2013 {dvs[2]}]'
                f'      established {ess[0]} [{ess[1]} \u2013 {ess[2]}]',
                ha='left', va='center', fontsize=FS_TICK, color=col)
    ax.set_ylim(-1.15, ROW * len(keys) - 0.05)
    ax.set_yticks([])
    year_axis(ax, 'Year')
    set_frame(ax, 'none')
    ax.spines['bottom'].set_visible(True)
    # key for the two density styles
    ax.text(XLIM[1] - 0.05, ROW * len(keys) - 0.45,
            'open = divergence from Nigerian relatives    filled = local establishment',
            ha='right', va='center', fontsize=FS_TICK, color='0.35')
    return medians, summary, nodes_xy


def density_side_label(ax):
    """'Posterior tMRCA' heading, to the right of the density block."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.06, 0.5, 'Posterior\ntMRCA', va='center', ha='left',
            fontsize=FS_ANN, weight='bold', color='0.15', linespacing=1.25)


# ================================================================== panel b
TREND_BW = 0.30            # kernel width in years, as in plot_lineage_trend.py
MIN_ESS = 2.0              # hatch intervals whose effective sample size is below this


def trend_frequencies(t, lineage, grid, bw=TREND_BW):
    """Kernel-weighted lineage frequency through time (Muller-style)."""
    present = [l for l in LIN_ORDER_TREND if l in set(lineage)]
    present += sorted(set(lineage) - set(present))
    W = np.exp(-0.5 * ((grid[:, None] - np.asarray(t)[None, :]) / bw) ** 2)
    ess = W.sum(axis=1)
    freqs = {l: (W * (np.asarray(lineage) == l)[None, :]).sum(axis=1)
             / np.maximum(ess, 1e-9) for l in present}
    return present, freqs, ess


LIN_ORDER_TREND = ['A', 'A.1', 'A.2', 'A.2.1', 'A.2.4', 'A.2.5', 'A.3',
                   'A.2.3', 'A.2.2', 'zoonotic']


def panel_trend(ax, dat):
    cur = dat['curated']
    ok = cur.t.notna()
    t, lineage = cur.t[ok].values, cur.lineage_plot[ok].values
    grid = np.linspace(XLIM[0], XLIM[1], 900)
    present, freqs, ess = trend_frequencies(t, lineage, grid)
    stack = np.vstack([freqs[l] for l in present])
    ax.stackplot(grid, stack, colors=[LIN_C.get(l, '#CCCCCC') for l in present],
                 edgecolor='white', linewidth=0.35, zorder=2)
    if 'zoonotic' in present:      # pale band: draw its lower edge so it reads
        below = stack[:present.index('zoonotic')].sum(axis=0)
        ax.plot(grid, below, color='#8C8C8C', lw=0.4, zorder=4)
    # mask intervals with too little data to support a frequency estimate
    weak = ess < MIN_ESS
    if weak.any():
        e = np.diff(weak.astype(int))
        starts = ([grid[0]] if weak[0] else []) + list(grid[1:][e == 1])
        ends = list(grid[1:][e == -1]) + ([grid[-1]] if weak[-1] else [])
        for a, b in zip(starts, ends):
            ax.axvspan(a, b, facecolor='white', alpha=0.72, lw=0, zorder=5)
            ax.axvspan(a, b, facecolor='none', edgecolor='#BBBBBB', hatch='///',
                       lw=0, zorder=6)
    # label the dominant bands in place, thickest first
    placed, n = [], len(grid)
    inset = max(1, int(0.10 * n))
    for l in sorted(present, key=lambda k: -freqs[k].max()):
        if l == 'zoonotic' or freqs[l].max() < 0.25:
            continue
        f = freqs[l]
        base = (np.vstack([freqs[k] for k in present[:present.index(l)]]).sum(axis=0)
                if present.index(l) else np.zeros_like(grid))
        good = np.ones(n, bool)
        good[:inset] = good[-inset:] = False
        good &= ~weak
        good &= f >= 0.12
        if not good.any():
            continue
        for i in np.argsort(np.where(good, f, -np.inf))[::-1]:
            if not good[i]:
                break
            x, y = grid[i], base[i] + f[i] / 2
            half = 0.012 * (grid[-1] - grid[0]) * len(l) * (FS_TICK / 25.0)
            if any(abs(x - px) < half + ph and abs(y - py) < 0.10
                   for px, py, ph in placed):
                continue
            dark = l in ('A.2.2', 'A.2.3', 'A.3', 'A.2.5')
            ax.text(x, y, l, ha='center', va='center', fontsize=FS_TICK,
                    weight='bold', color=('white' if dark else '#222222'), zorder=8)
            placed.append((x, y, half))
            break
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('Lineage frequency', fontsize=FS_BASE, labelpad=2)
    year_axis(ax, 'Year')
    ax.tick_params(axis='y', length=2.5, pad=1.5, labelsize=FS_TICK)
    set_frame(ax, 'open')
    return present, freqs, ess


def trend_key(ax, present):
    """Lineage key for panel b, in the right-hand column (keeps b's axis width)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.06, 0.985, 'Lineage', fontsize=FS_ANN, weight='bold', va='top',
            ha='left', color='0.15', transform=ax.transAxes)
    y, dy = 0.862, 0.082
    for l in reversed(present):
        ax.add_patch(Rectangle((0.06, y - dy * 0.30), 0.22, dy * 0.60,
                               facecolor=LIN_C.get(l, '#CCCCCC'), edgecolor='0.55',
                               linewidth=0.2, transform=ax.transAxes, clip_on=False))
        ax.text(0.34, y, 'Zoonotic' if l == 'zoonotic' else l, fontsize=FS_TICK,
                va='center', ha='left', color='0.15', transform=ax.transAxes)
        y -= dy


# ================================================================== assemble
def build(dat, outfile='Figure4_MPXV_exports.png', figsize=(7.2, 9.2)):
    global XLIM
    root = dat['root']
    XLIM = (2015.75, 2026.45)
    fig = plt.figure(figsize=figsize, facecolor='w')
    # the tree and the density block abut (hspace=0) so they read as one panel
    # on one time axis: only the density block carries year ticks and labels
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.145],
                          height_ratios=[2.26, 0.64],
                          wspace=0.012, hspace=0.14,
                          left=0.072, right=0.988, top=0.978, bottom=0.048)
    gl = gs[0, 0].subgridspec(2, 1, height_ratios=[1.70, 0.56], hspace=0.0)
    gr = gs[0, 1].subgridspec(2, 1, height_ratios=[1.70, 0.56], hspace=0.0)
    axT = fig.add_subplot(gl[0]); axL = fig.add_subplot(gr[0])
    axD = fig.add_subplot(gl[1]); axDl = fig.add_subplot(gr[1])
    axB = fig.add_subplot(gs[1, 0]); axBk = fig.add_subplot(gs[1, 1])

    med, summary, node_xy = panel_density(axD, dat)
    order = panel_tree(axT, dat, med, node_xy)
    lineage_bars(axL, dat, order)
    tip_key(axT, dat)
    density_side_label(axDl)
    present, freqs, ess = panel_trend(axB, dat)
    trend_key(axBk, present)

    for ax, lab in ((axT, 'a'), (axB, 'b')):
        ax.text(-0.062, 1.004, lab, transform=ax.transAxes, fontsize=11,
                weight='bold', va='bottom', ha='left')
    fig.savefig(outfile)
    return fig, summary


def main(data_dir=None, out_stem='Figure4_MPXV_exports'):
    d = os.path.abspath(data_dir or _HERE).rstrip('/') + '/'
    apply_figure_style(sizes=(FS_BASE, FS_ANN, FS_TICK))
    dat = load_data4(d)
    fig, summary = build(dat, out_stem + '.png')
    pd.DataFrame(summary).to_csv('tmrca_summary.csv', index=False)
    for ext in ('pdf', 'svg'):
        fig.savefig(f'{out_stem}.{ext}')
    print(f'wrote {out_stem}.png/.pdf/.svg and tmrca_summary.csv from {d}')
    return fig


if __name__ == '__main__':
    main(*sys.argv[1:3])
