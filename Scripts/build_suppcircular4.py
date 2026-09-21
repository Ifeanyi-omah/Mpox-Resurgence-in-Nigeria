"""Supplementary figure for the MPXV manuscript.

Panel A  the global clade IIb maximum-likelihood tree (All+MPOX+CladeIIb.pruned.tree,
         9,268 genomes), branch lengths in mutations, no per-branch substitution
         markers, tips coloured under the collapsed rule: Nigeria keeps its two
         greys, Sierra Leone keeps its own colour, every other country collapses
         to its continent, in the circular panel's palette. The West African
         A.2.2 / A.2.3 / G.1 block that panel B expands is ringed with a broken
         square.
Panel B  the subsampled circular tree with the central world map
         (673+pruned.tree, 988 genomes, Sierra Leonean G.1 thinned), drawn by
         mpox_circular.py at its original geometry, with its original
         per-country African legend and the continent legend retained.

All text is written as text (pdf.fonttype 42, svg.fonttype 'none'), never outlined.
"""
import os
import sys
import collections

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import mpox_circular as M          # noqa: E402
import panelA as A                 # noqa: E402

PRINT_SIZE = True            # False reverts to the 54 x 36 in working canvas

TREE_A = os.path.join(BASE, 'global_full.tree')   # All+MPOX+CladeIIb.pruned.tree
TREE_B = os.path.join(BASE, 'panelB.nexus')       # 673+tree.nexus
HIGHLIGHT = os.path.join(BASE, 'Nigeria_new_genomes.csv')
NEXTCLADE = os.path.join(BASE, 'nextclade_lineages.tsv')
OUT = os.path.join(BASE, 'SupplementaryFigure_global_and_circular')

# ── canvas ─────────────────────────────────────────────────────────────
# Drawn at final print size: 183 mm (Nature double column) wide, so every point
# size below is the size it prints at. Panel B's geometry was specified for a
# ~28 in canvas, so mpox_circular.set_size_scale() rescales its markers and
# linewidths to the panel's real width and the type ladder is set explicitly.
if PRINT_SIZE:
    FIGSIZE = (7.2, 6.3)                     # 183 x 160 mm
    BOX_A = (0.048, 0.098, 0.300, 0.862)     # tall left column: the 9,268-tip tree
    BOX_B = (0.360, 0.040, 0.625, 0.920)     # right column: circular tree + map
    FS_BASE, FS_ANN, FS_TICK = 8.0, 7.0, 6.0
    FS_LETTER = 10.0
    PANEL_B_SCALE = 0.160
    TIP_S, BRANCH_LW, TERMINAL_MULT = 5.0, 0.18, 1.7
    STRIP_W, ZOOM_LW = 0.0, 0.6
else:
    FIGSIZE = (54.0, 36.0)
    BOX_A = (0.040, 0.072, 0.300, 0.900)
    BOX_B = (0.365, 0.055, 0.625, 0.900)
    FS_BASE, FS_ANN, FS_TICK = 45.0, 45.0, 38.0
    FS_LETTER = 66.0
    PANEL_B_SCALE = 1.0
    TIP_S, BRANCH_LW, TERMINAL_MULT = 26.0, 1.5, 1.9
    STRIP_W, ZOOM_LW = 0.022, 2.6

FS_LEG = FS_ANN      # panel A key text
FS_LAB = FS_BASE     # axis label
LEG_MS = FS_ANN * 0.62   # panel A key marker, sized off the key's own type
A_TIP_DOTS = True    # a dot on every tip, as in the reference layout; they merge where the
                     # tree is dense, which is the intended reading of tip density

ZOOM_LINEAGES = ('A.2.2', 'A.2.3', 'G.1')   # what panel B expands
ZOOM_COLOUR = '#1F5C99'


def leaf_set(node):
    out, stack = [], [node]
    while stack:
        n = stack.pop()
        if n.branchType == 'leaf':
            out.append(n)
        else:
            stack.extend(n.children)
    return out


def mrca_of(tips):
    want = {id(k) for k in tips}
    node = tips[0]
    while node.parent is not None:
        node = node.parent
        if want <= {id(k) for k in leaf_set(node)}:
            return node
    return node


def load_tree_a():
    """The global tree as baltic lays it out, in substitutions.

    Deliberately NOT sorted: `sortBranches` reorders each node's children by
    clade size, which ladders the basal splits and leaves several full-height
    vertical connectors stacked beside each other. Keeping the file's own child
    order leaves the global ingroup hanging off a single connector, as in the
    reference layout.
    """
    import baltic as bt
    t = bt.loadNewick(TREE_A, absoluteTime=False)
    t.traverse_tree()
    for k in t.Objects:
        k.length = (k.length or 0.0) * M.MPOX_ALN_LENGTH
    t.traverse_tree()
    M.safe_draw_tree(t)
    return t, [k for k in t.Objects if k.branchType == 'leaf']


def draw_panel_a(ax, highlight, lineage_map):
    """The global tree, drawn by baltic. Returns (tally, zoom_box, n_tips)."""
    tree, leaves = load_tree_a()
    tipcol = {id(k): A.collapsed_colour(k.name, highlight) for k in leaves}
    cols = [tipcol[id(k)] for k in leaves]

    xa = lambda k: k.height          # baltic's own x attribute: substitutions
    tree.plotTree(ax, x_attr=xa, zorder=2,
                  colour=lambda k: tipcol.get(id(k), '#6E6E6E'),
                  width=lambda k: (BRANCH_LW * TERMINAL_MULT if id(k) in tipcol
                                   else BRANCH_LW))
    if A_TIP_DOTS:
        xr = max(k.height for k in leaves) or 1.0
        tree.plotPoints(ax, x_attr=xa, zorder=4,
                        colour=lambda k: tipcol[id(k)],
                        size=lambda k: max(TIP_S * 0.40,
                                           TIP_S * (1.0 - 0.25 * k.height / xr)))
    # this study's genomes drawn on top, ringed, so they can be located
    hi = [k for k in leaves if k.name in highlight]
    ax.scatter([k.height for k in hi], [k.y for k in hi], s=TIP_S * 1.8,
               facecolors=M.NIGERIA_NEW_GREY, edgecolors='w',
               linewidths=BRANCH_LW * 1.6, zorder=5)

    xmax = max(k.height for k in leaves)
    ymax = max(k.y for k in leaves)

    if STRIP_W > 0:
        rgb = np.array([mpl.colors.to_rgb(c) for c in cols])[:, None, :]
        ax.imshow(rgb, aspect='auto', interpolation='nearest', origin='lower',
                  extent=(xmax * 1.015, xmax * (1.015 + STRIP_W), -0.5, ymax + 0.5),
                  zorder=6, clip_on=False)

    ax.set_xlim(-xmax * 0.02, xmax * (1.02 + STRIP_W))
    ax.set_ylim(ymax * 1.012, -ymax * 0.012)   # West African block at the top
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='x', labelsize=FS_TICK, length=2.2, width=0.5, pad=2,
                   direction='out')
    ax.set_xlabel('Substitutions from the root of clade IIb', fontsize=FS_LAB,
                  labelpad=3)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_bounds(0, 100)

    # the block panel B expands, ringed with a broken square
    lin = {k.name: M._lineage_of_tip(k.name, lineage_map) for k in leaves}
    core = [k for k in leaves if lin[k.name] in ZOOM_LINEAGES]
    node = mrca_of(core)
    clade = leaf_set(node)
    y0, y1 = min(k.y for k in clade), max(k.y for k in clade)
    x0 = node.parent.height if node.parent is not None else 0.0
    x1 = max(k.height for k in clade)
    pad_y = ymax * 0.006
    pad_x = xmax * 0.012
    ax.add_patch(Rectangle((x0 - pad_x, y0 - pad_y),
                           (x1 - x0) + 2 * pad_x, (y1 - y0) + 2 * pad_y,
                           transform=ax.transData, facecolor='none',
                           edgecolor=ZOOM_COLOUR, linewidth=ZOOM_LW,
                           linestyle=(0, (4, 2.5)), zorder=8, clip_on=False))
    ax.annotate('expanded in B', xy=(x1 + pad_x, y0 - 2.2 * pad_y),
                xytext=(0, 1.5), textcoords='offset points', color=ZOOM_COLOUR,
                ha='right', va='bottom', fontsize=FS_ANN, zorder=9,
                annotation_clip=False)

    tally = collections.Counter(A.colour_category(k.name, highlight) for k in leaves)
    return tally, (x0, y0, x1, y1, len(clade)), len(leaves)


# short key labels: the per-category counts live in the caption, not the figure
KEY_LABELS = {
    'Nigeria \u2013 this study': 'Nigeria, this study',
    'Nigeria \u2013 previously published': 'Nigeria, published',
    'Sierra Leone': 'Sierra Leone',
    'Africa, other countries': 'Africa, other',
    'Europe': 'Europe',
    'Asia': 'Asia',
    'Americas': 'Americas',
    'Other / unknown': 'Other/unknown',
}


def panel_a_key(ax, x=0.020, y=0.585, dy=0.0245):
    """Swatch-first key in the panel's own whitespace, frameless."""
    for i, (lab, col) in enumerate(A.PANEL_A_KEY):
        yy = y - i * dy
        ax.plot([x], [yy], marker='s', ms=LEG_MS, mfc=col, mec='none',
                transform=ax.transAxes, clip_on=False, zorder=9)
        ax.text(x + 0.055, yy, KEY_LABELS[lab], transform=ax.transAxes,
                va='center', ha='left', fontsize=FS_LEG, color='0.05',
                clip_on=False, zorder=9)


def g1_names_from(treefile, lineage_map):
    import baltic as bt
    t = bt.loadNewick(treefile, absoluteTime=False)
    t.traverse_tree()
    return {k.name for k in t.Objects if k.branchType == 'leaf'
            and M._lineage_of_tip(k.name, lineage_map) == 'G.1'}


def build():
    # type ladder and line weights for the printed page; panel B's markers and
    # linewidths are rescaled from its 28 in design to the panel's real width
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FS_ANN, 'axes.labelsize': FS_BASE, 'axes.titlesize': FS_BASE,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_ANN, 'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 400, 'figure.dpi': 400,
    })
    M.set_size_scale(PANEL_B_SCALE, fonts={
        'SCALE_TEXT_SIZE': FS_TICK, 'LEGEND_TEXT_SIZE': FS_ANN,
        'LINEAGE_TEXT_SIZE': FS_TICK, 'LINEAGE_LIST_FONT_SIZE': FS_TICK,
    })

    highlight = set(pd.read_csv(HIGHLIGHT)['seqName'].astype(str).str.strip())
    lineage_map = M._load_nextclade_lineages(NEXTCLADE)

    fig = plt.figure(figsize=FIGSIZE, facecolor='w')
    ax_a = fig.add_axes(BOX_A)
    tally, zoom, n_a = draw_panel_a(ax_a, highlight, lineage_map)
    panel_a_key(ax_a)

    g1 = g1_names_from(TREE_B, lineage_map)
    M.make_circular_global_figure(
        treefile=TREE_B, outfile=None, highlight_csv=HIGHLIGHT,
        nextclade_tsv=NEXTCLADE, sl_keep=M.SL_KEEP, aln_length=M.MPOX_ALN_LENGTH,
        g1_names=g1, fig=fig, box=BOX_B, save=False, drop_above_mutations=200,
        map_width=M.MAP_WIDTH, circStart=0.83, circFrac=0.97, gap_frac=M.GAP_FRAC)

    fig.text(0.008, 0.992, 'A', fontsize=FS_LETTER, fontweight='bold',
             ha='left', va='top')
    fig.text(BOX_B[0] - 0.020, 0.992, 'B', fontsize=FS_LETTER,
             fontweight='bold', ha='left', va='top')

    for ext, kw in [('pdf', {}), ('svg', {}), ('png', {'dpi': 400})]:
        fig.savefig(f'{OUT}.{ext}', facecolor='w', **kw)
        print('saved', f'{OUT}.{ext}')

    print(f'panel A: {n_a:,} tips; zoom block x {zoom[0]:.1f}-{zoom[2]:.1f} mut, '
          f'tips {zoom[4]:,} (y {zoom[1]:.0f}-{zoom[3]:.0f})')
    for lab, _ in A.PANEL_A_KEY:
        print(f'   {lab}: {tally.get(lab, 0)}')
    print(f'panel B: G.1 tips supplied {len(g1)}')
    return fig


if __name__ == '__main__':
    build()
