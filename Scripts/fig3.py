"""Figure 3: zoonotic MPXV phylogeny, posterior node ages and sampling map.

Panels
    a  time-scaled MCC tree of the 26 zoonotic and Cameroonian genomes, with
       the human epidemic collapsed to a single triangle; focal nodes ringed
    b  posterior densities for seven focal node ages, on panel a's time axis
    c  Nigeria and Cameroon, states filled by geopolitical zone, sampled
       states outlined and labelled, sampling sites marked

Node ages come from the posterior tree file except the root and the APOBEC3
transition, which are logged parameters. Usage:

    python fig3.py [data_dir] [out_stem]
"""
import os
import re
import sys
import json
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _HERE)
import mpxv4 as m4                         # tree reader, calibration, HPD
from mpxvfig import ZONES, ZONE_C, STATE_C, CMR_UNSAMPLED, CMR_SAMPLED, \
    CMR_SAMPLED_REGIONS, geo_patches       # study palette and map patches

FS_BASE, FS_ANN, FS_TICK = 10, 9, 8

STATE_PRETTY = {'AkwaIbom': 'Akwa Ibom', 'CrossRiver': 'Cross River',
                'Imo': 'Imo', 'Abia': 'Abia', 'Ondo': 'Ondo', 'Edo': 'Edo'}
CAM_C = CMR_SAMPLED
HUMAN_C = '#3F4A4F'

ROOT_LABEL_FULL = 'Root: 1971 Abia genome diverges'
ROOT_LABEL_PRUNED = 'Border and south-east lineages diverge'


def focal_spec(pruned=False):
    """Focal nodes, label -> (source, colour); 'log:' names a log column.

    The APOBEC3 transition is deliberately absent: `apobec3.stem.proportion` is
    fixed at 1.000 in the XML, so that parameter is pinned to the top of the
    human stem and is not an estimate original to this analysis.
    """
    root = ((ROOT_LABEL_PRUNED, 'tree:ingroup', '#7B1E2B') if pruned else
            (ROOT_LABEL_FULL, 'log:age(root)', '#7B1E2B'))
    return [
        root,
        ('Nigeria\u2013Cameroon border lineage',       'tree:border',   '#966919'),
        ('South-east Nigeria lineage',                 'tree:se_human', '#097969'),
        ('Akwa Ibom diverges from South-West Cameroon', 'tree:ak4_stem', '#C19A6B'),
        ('Human epidemic MRCA',                        'tree:human',    '#000000'),
        ('Akwa Ibom cluster MRCA',                     'tree:ak4',      '#8C6A3F'),
    ]


FOCAL = focal_spec()
XLIM = (1966.0, 2030.0)
XLIM_PRUNED = (1990.0, 2030.0)
YEAR_TICKS = list(range(1970, 2031, 10))
YEAR_TICKS_PRUNED = list(range(1990, 2031, 10))
BAND_STEP = 10          # alternating decade shading, as in Figure 4's year bands


def apply_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': FS_BASE, 'axes.linewidth': 0.6,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 2.5, 'xtick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'figure.dpi': 200, 'savefig.dpi': 400,
        'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def year_bands(ax, xlim, ticks, colour='#F0F0F0', step=BAND_STEP):
    """Alternating translucent decade shading behind the data, and a white
    hairline on every labelled year, so the tree and the densities above and
    below it read against one time scale."""
    x0, x1 = xlim
    y = int(np.floor(x0 / step) * step)
    k = 0
    while y < x1:
        if k % 2 == 0:
            ax.axvspan(max(y, x0), min(y + step, x1), color=colour, lw=0, zorder=0)
        y += step
        k += 1
    for t in ticks:
        if x0 <= t <= x1:
            ax.axvline(t, color='white', lw=0.6, zorder=1)


def year_axis(ax, xlim, ticks, labels=True, label=None, minor_step=5):
    """Identical time axis on the tree and the density block."""
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) if labels else '' for t in ticks], fontsize=FS_TICK)
    minors = [t for t in range(int(xlim[0]), int(xlim[1]) + 1, minor_step)
              if t not in ticks]
    ax.set_xticks(minors, minor=True)
    ax.tick_params(axis='x', length=(2.8 if labels else 0), pad=2.0,
                   labelbottom=labels)
    ax.tick_params(axis='x', which='minor', length=(1.5 if labels else 0))
    if label:
        ax.set_xlabel(label, fontsize=FS_ANN, labelpad=2.0)


def halo(artist, lw=1.4, fg='white'):
    artist.set_path_effects([pe.withStroke(linewidth=lw, foreground=fg)])
    return artist


def state_of(label):
    for raw, pretty in STATE_PRETTY.items():
        if re.search(raw, label, re.I) or re.search(raw.replace('AkwaIbom', 'Akwa-Ibom'),
                                                    label, re.I):
            return pretty
    if 'cameroon' in label.lower():
        return 'Cameroon'
    return None


def tip_colour(label):
    st = state_of(label)
    if st == 'Cameroon':
        return CAM_C
    key = {v: k for k, v in STATE_PRETTY.items()}.get(st)
    return STATE_C.get(key, '0.4')


def pretty_tip(label):
    """'Akwa Ibom | 2024-09-03' from a raw tip label."""
    st = state_of(label) or label.split('|')[0]
    md = re.search(r'(\d{4}-\d{2}-\d{2})', label)
    if md:
        return f'{st} | {md.group(1)}'
    my = re.search(r'\b(\d{4})\b', label)
    return f'{st} | {my.group(1)}' if my else st


# --------------------------------------------------------------- tree layout
def display_layout(root, collapse_node, keep):
    """y positions treating `collapse_node` as one leaf and dropping tips not
    in `keep`. Returns (rows, ypos) where rows is the drawing order."""
    def kids(nd):
        if nd is collapse_node:
            return []
        out = []
        for ch in nd.children:
            if ch is collapse_node:
                out.append(ch)
            elif ch.children:
                if any(l.name in keep or l is collapse_node
                       for l in m4.leaves(ch)) or _has(ch, collapse_node):
                    out.append(ch)
            elif ch.name in keep:
                out.append(ch)
        return out

    rows, ypos = [], {}
    counter = [0]

    def walk(nd):
        ch = kids(nd)
        if not ch:
            ypos[id(nd)] = counter[0]
            rows.append(nd)
            counter[0] += 1
            return ypos[id(nd)]
        ys = [walk(c) for c in ch]
        ypos[id(nd)] = 0.5 * (min(ys) + max(ys))
        return ypos[id(nd)]

    walk(root)
    return rows, ypos, kids


def _has(nd, target):
    st = [nd]
    while st:
        x = st.pop()
        if x is target:
            return True
        st.extend(x.children)
    return False


def ladderize(root, collapse_node, keep, deepest_first=True):
    """Order children so the figure reads outgroup-first, human clade last."""
    def n_disp(nd):
        if nd is collapse_node:
            return 1
        if not nd.children:
            return 1 if nd.name in keep else 0
        return sum(n_disp(c) for c in nd.children)

    st = [root]
    while st:
        nd = st.pop()
        if nd is collapse_node:
            continue
        nd.children.sort(key=lambda c: (_has(c, collapse_node), n_disp(c)),
                         reverse=not deepest_first)
        st.extend(nd.children)


# ================================================================== panel a
def panel_a(ax, axl, root, collapse_node, keep, C, focal_nodes, brackets,
            weak_pp=0.95, xlim=None, ticks=None):
    """Rectangular time tree in `ax`; tip labels and clade brackets in `axl`.

    Splitting the labels into their own axes keeps panel a's x-axis purely
    calendar time, so panel b can share it exactly.
    """
    xlim = xlim or XLIM
    ticks = ticks or YEAR_TICKS
    ladderize(root, collapse_node, keep)
    rows, ypos, kids = display_layout(root, collapse_node, keep)
    n = len(rows)
    year_bands(ax, xlim, ticks)

    segs = []
    stack = [root]
    while stack:
        nd = stack.pop()
        ch = kids(nd)
        if not ch:
            continue
        ys = [ypos[id(c)] for c in ch]
        segs.append([(nd.x, min(ys)), (nd.x, max(ys))])          # vertical
        for c in ch:
            segs.append([(nd.x, ypos[id(c)]), (c.x, ypos[id(c)])])   # horizontal
            stack.append(c)
    segs.append([(xlim[0] + 0.8, ypos[id(root)]), (root.x, ypos[id(root)])])  # root stub
    ax.add_collection(LineCollection(segs, colors='0.25', linewidths=0.55, zorder=2))

    # tips: markers in the tree axes, text in the label strip
    for nd in rows:
        y = ypos[id(nd)]
        if nd is collapse_node:
            tipmax = max(l.x for l in m4.leaves(nd))
            ax.add_patch(Polygon([[nd.x, y], [tipmax, y - 0.7], [tipmax, y + 0.7]],
                                 closed=True, facecolor=HUMAN_C, edgecolor='0.2',
                                 linewidth=0.4, alpha=0.85, zorder=3))
            axl.text(0.012, y, f'human epidemic clade (n = {len(m4.leaves(nd))})',
                     va='center', ha='left', fontsize=FS_TICK, color=HUMAN_C,
                     fontweight='bold')
            continue
        c = tip_colour(nd.name)
        ax.plot([nd.x], [y], 'o', ms=3.0, mfc=c, mec='0.2', mew=0.35, zorder=4)
        axl.text(0.012, y, pretty_tip(nd.name), va='center', ha='left',
                 fontsize=FS_TICK, color='0.12')

    # posterior support, shown only where it is not effectively certain
    for nd in rows + [r for r in _all_internal(root, collapse_node, kids)]:
        if nd is collapse_node or not kids(nd):
            continue
        p = (getattr(nd, 'ann', {}) or {}).get('posterior')
        if p is not None and p < weak_pp:
            halo(ax.text(nd.x - 0.6, ypos[id(nd)] + 0.02, f'{p:.2f}', va='center',
                         ha='right', fontsize=FS_TICK - 1.0, color='0.45'), lw=1.2)

    for lab, node, col in focal_nodes:
        y = ypos.get(id(node))
        if y is not None:
            ax.plot([node.x], [y], 'o', ms=5.4, mfc='none', mec=col, mew=1.0, zorder=6)

    # clade brackets in the label strip, placed past the measured right edge of
    # the tip labels so they survive a change of font size
    axl.set_xlim(0, 1)
    axl.set_ylim(n - 0.4, -1.1)
    axl.figure.canvas.draw()
    rend = axl.figure.canvas.get_renderer()
    w = axl.get_window_extent(rend)
    right = max((t.get_window_extent(rend).x1 for t in axl.texts), default=w.x0)
    xb = min((right - w.x0) / max(w.width, 1) + 0.035, 0.55)
    drawn = []
    for lab, node, col, _xoff in brackets:
        lv = [ypos[id(r)] for r in rows if _has(node, r)]
        if not lv:
            continue
        y0, y1 = min(lv), max(lv)
        line, = axl.plot([xb, xb], [y0 - 0.3, y1 + 0.3], color=col, lw=1.2,
                         solid_capstyle='butt', zorder=5, clip_on=False)
        txt = axl.text(xb + 0.025, 0.5 * (y0 + y1), lab, va='center', ha='left',
                       fontsize=FS_TICK, color=col, fontweight='bold',
                       linespacing=1.2)
        drawn.append((line, txt, y0, y1))
    # second pass: pull the brackets left by however much the widest label
    # overruns the strip, so the text never leaves the canvas
    axl.figure.canvas.draw()
    over = max((t.get_window_extent(rend).x1 - w.x1 for _l, t, _a, _b in drawn),
               default=0.0)
    if over > 0:
        shift = over / max(w.width, 1) + 0.01
        for line, txt, y0, y1 in drawn:
            line.set_xdata([xb - shift, xb - shift])
            txt.set_x(xb - shift + 0.025)

    ax.set_ylim(n - 0.4, -1.1)
    ax.set_yticks([])
    year_axis(ax, xlim, ticks, labels=False)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    axl.axis('off')
    return rows, ypos


# ================================================================== panel b
def kde(v, grid, bw=0.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    from scipy.stats import gaussian_kde
    if len(v) < 3 or np.std(v) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(v, bw_method=bw).evaluate(grid)


def panel_b(ax, series, order=None, row_h=1.0, xlim=None, ticks=None,
            calendar=True):
    """One filled posterior density per focal node, ordered in time.

    `series` maps label -> (values, colour). Each row is scaled to a common
    height so narrow posteriors stay visible next to broad ones.
    """
    xlim = xlim or XLIM
    ticks = ticks or YEAR_TICKS
    labs = order or sorted(series, key=lambda k: np.median(series[k][0]))
    year_bands(ax, xlim, ticks)
    summary = []
    for i, lab in enumerate(labs):
        v, col = series[lab]
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        # each curve is evaluated on its own support and trimmed to where it is
        # non-negligible, so no row carries a flat baseline across the panel
        pad = 0.35 * (v.max() - v.min() + 1e-6)
        grid = np.linspace(max(v.min() - pad, xlim[0]),
                           min(v.max() + pad, xlim[1]), 600)
        y = kde(v, grid)
        if y.max() > 0:
            keep = y > 0.006 * y.max()
            if keep.any():
                j0, j1 = int(np.argmax(keep)), len(keep) - int(np.argmax(keep[::-1]))
                grid, y = grid[j0:j1], y[j0:j1]
            y = y / y.max() * (row_h * 0.86)
        base = i * row_h
        med = float(np.median(v))
        lo, hi = m4.hpd(v)
        ax.fill_between(grid, base, base + y, facecolor=col, alpha=0.40,
                        edgecolor='none', zorder=3)
        ax.plot(grid, base + y, color=col, lw=0.9, zorder=4)
        ax.plot([med, med], [base, base + np.interp(med, grid, y)], color=col,
                lw=1.1, zorder=5)
        ax.plot([lo, hi], [base - 0.06 * row_h] * 2, color=col, lw=1.6,
                solid_capstyle='butt', zorder=5)
        if calendar:
            ax.text(hi + 1.0, base + 0.60 * row_h, lab, va='center', ha='left',
                    fontsize=FS_TICK, color=col, fontweight='bold', clip_on=False)
            ax.text(hi + 1.0, base + 0.20 * row_h,
                    f'{m4.to_calendar(med)}   95% HPD '
                    f'{m4.to_calendar(lo)} to {m4.to_calendar(hi)}',
                    va='center', ha='left', fontsize=FS_TICK - 1.0, color=col,
                    clip_on=False)
        else:
            ax.text(hi + 1.0, base + 0.34 * row_h,
                    f'{lab}: {med:.1f} (95% HPD {lo:.1f}\u2013{hi:.1f})',
                    va='center', ha='left', fontsize=FS_TICK, color=col,
                    fontweight='bold', clip_on=False)
        summary.append(dict(node=lab, median=round(med, 2),
                            median_date=m4.to_calendar(med),
                            hpd_lo=round(lo, 2), hpd_lo_date=m4.to_calendar(lo),
                            hpd_hi=round(hi, 2), hpd_hi_date=m4.to_calendar(hi),
                            n_samples=len(v)))
    ax.set_ylim(-0.30 * row_h, len(labs) * row_h)
    ax.set_yticks([])
    year_axis(ax, xlim, ticks, labels=True, label='Year')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.text(-0.016, 0.5, 'posterior\nnode age', transform=ax.transAxes, rotation=90,
            va='center', ha='center', fontsize=FS_ANN, color='0.15', linespacing=1.2)
    return labs, pd.DataFrame(summary)


def guide_lines(ax_tree, ax_dens, focal_nodes, ypos, labs, series, row_h=1.0):
    """Dashed droplines from each ringed node in panel a to its density in b."""
    for lab in labs:
        col = series[lab][1]
        node = next((nd for l, nd, c in focal_nodes if l == lab), None)
        if node is None or id(node) not in ypos:
            continue
        x = node.x
        ax_tree.plot([x, x], [ypos[id(node)], ax_tree.get_ylim()[0]], color=col,
                     lw=0.4, ls=(0, (2.2, 1.6)), alpha=0.65, zorder=1)
        i = labs.index(lab)
        ax_dens.plot([x, x], [ax_dens.get_ylim()[1], i * row_h], color=col,
                     lw=0.4, ls=(0, (2.2, 1.6)), alpha=0.65, zorder=1)


# ================================================================== panel c
def _rings(feat):
    g = feat['geometry']
    if g['type'] == 'Polygon':
        return [g['coordinates'][0]]
    return [p[0] for p in g['coordinates']]


def _inside(feat, lon, lat):
    from matplotlib.path import Path
    return any(Path(np.asarray(r)[:, :2]).contains_point((lon, lat))
               for r in _rings(feat))


def _centroid(feat):
    r = max(_rings(feat), key=len)
    a = np.asarray(r)[:, :2]
    return a[:, 0].mean(), a[:, 1].mean()


def fix_coords(df, feats, name_col, lat_col='lat', lon_col='long'):
    """Return (lon, lat) per row, swapping the pair when the named polygon
    contains the transposed point but not the tabulated one.

    The curated table stores lat and long the other way round for every
    Cameroonian row and for at least one Nigerian row, so a fixed per-country
    rule is not enough; each point is tested against its own admin polygon.
    """
    by_name = {}
    for f in feats:
        by_name[f['properties']['NAME_1'].lower().replace(' ', '').replace('-', '')] = f
    out, swapped = [], []
    for _, r in df.iterrows():
        a, b = float(r[lat_col]), float(r[lon_col])
        nm = str(r[name_col]).lower().replace(' ', '').replace('-', '')
        f = by_name.get(nm)
        if f is None:
            out.append((b, a))                    # assume (lon, lat) = (long, lat)
            swapped.append(None)
            continue
        if _inside(f, b, a):
            out.append((b, a))
            swapped.append(False)
        elif _inside(f, a, b):
            out.append((a, b))
            swapped.append(True)
        else:
            cx, cy = _centroid(f)                 # neither orientation lands inside
            out.append((cx, cy))
            swapped.append('centroid')
    return out, swapped


def panel_c(ax, nga, cmr, sites, xlim=(2.4, 16.4), ylim=(3.6, 14.0),
            label_states=('Ondo', 'Edo', 'Imo', 'Abia', 'AkwaIbom', 'CrossRiver'),
            state_label_xy=None):
    """Nigeria by geopolitical zone, Cameroon by sampled region, sites marked."""
    zone_of = {s: z for z, ss in ZONES.items() for s in ss}
    cam = geo_patches(cmr['features'],
                      lambda n: CMR_SAMPLED if n in CMR_SAMPLED_REGIONS else CMR_UNSAMPLED,
                      lw=0.25, ec='white')
    ax.add_collection(cam)
    nig = geo_patches(nga['features'],
                      lambda n: ZONE_C.get(zone_of.get(n.replace(' ', '')), '0.85'),
                      lw=0.25, ec='white')
    ax.add_collection(nig)
    sampled = [f for f in nga['features']
               if f['properties']['NAME_1'].replace(' ', '') in label_states]
    out = geo_patches(sampled, lambda n: 'none', lw=0.7, ec='0.1')
    out.set_zorder(6)
    ax.add_collection(out)
    if state_label_xy:
        for nm, (lx, ly) in state_label_xy.items():
            f = next(f for f in nga['features']
                     if f['properties']['NAME_1'].replace(' ', '') == nm)
            cx, cy = _centroid(f)
            ax.plot([cx, lx], [cy, ly], color='0.25', lw=0.35, zorder=7)
            ax.plot([cx], [cy], 'o', ms=1.1, color='0.15', zorder=8)
            halo(ax.text(lx, ly, STATE_PRETTY.get(nm, nm), fontsize=FS_TICK - 0.5,
                         ha='center', va='center', color='0.05', zorder=9))
    for (lon, lat), col in sites:
        # Cameroonian sites sit on the red 'sampled region' fill, so they are
        # drawn as white pins ringed in the same red rather than red on red
        # white pins ringed in the tip colour: the zone fills are mid-tone
        # oranges and greens, so filled dots in those same hues disappear
        ax.plot([lon], [lat], 'o', ms=3.4, mfc='white', mec=col, mew=1.0,
                zorder=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # scale bar, 200 km at this latitude
    km = 200.0 / (111.32 * np.cos(np.radians(0.5 * (ylim[0] + ylim[1]))))
    x0, y0 = xlim[1] - km - 0.6, ylim[0] + 0.35
    ax.plot([x0, x0 + km], [y0, y0], color='0.1', lw=1.1, solid_capstyle='butt')
    ax.text(x0 + km / 2, y0 + 0.22, '200 km', ha='center', va='bottom',
            fontsize=FS_TICK - 0.5, color='0.1')
    return ax


def zone_legend(ax, ncol=2):
    """Legend for the zone fills -- requested so readers can place 'south-east'."""
    handles = [Patch(facecolor=ZONE_C[z], edgecolor='white', linewidth=0.3, label=z)
               for z in ZONES]
    handles += [Patch(facecolor=CMR_SAMPLED, edgecolor='white', linewidth=0.3,
                      label='Cameroon, sampled region'),
                Patch(facecolor=CMR_UNSAMPLED, edgecolor='white', linewidth=0.3,
                      label='Cameroon, not sampled')]
    ax.axis('off')
    ax.legend(handles=handles, loc='upper left', fontsize=FS_TICK, ncol=ncol,
              frameon=False, handlelength=1.0, handleheight=0.85,
              columnspacing=0.8, handletextpad=0.35, labelspacing=0.42,
              title='Nigerian geopolitical zone', title_fontsize=FS_TICK,
              alignment='left')
    return ax


def _all_internal(root, collapse_node, kids):
    out, st = [], [root]
    while st:
        nd = st.pop()
        if nd is collapse_node:
            continue
        if kids(nd):
            out.append(nd)
            st.extend(kids(nd))
    return out



# ================================================================== assembly
def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=FS_BASE + 1, fontweight='bold', va='top', ha='left')


def build(dat, figsize=(7.2, 9.7)):
    """Assemble the figure at Nature double-column width.

    Panel a is the tree and the posterior node ages together: the two axes abut
    with no gap and share one time scale, the tree drawn without tick labels so
    the density block below it carries the only year axis. Panel b is the map.
    """
    apply_style()
    xlim, ticks = dat['xlim'], dat['ticks']
    fig = plt.figure(figsize=figsize, facecolor='w')
    axT = fig.add_axes([0.058, 0.485, 0.502, 0.495])
    axL = fig.add_axes([0.560, 0.485, 0.432, 0.495])
    axB = fig.add_axes([0.058, 0.330, 0.502, 0.155])
    axM = fig.add_axes([0.010, 0.000, 0.585, 0.275])
    axK = fig.add_axes([0.610, 0.165, 0.385, 0.110])
    axK2 = fig.add_axes([0.610, 0.020, 0.385, 0.135])

    rows, ypos = panel_a(axT, axL, dat['root'], dat['human_node'], dat['keep'],
                         dat['C'], dat['focal'], dat['brackets'],
                         xlim=xlim, ticks=ticks)
    labs, summary = panel_b(axB, dat['series'], xlim=xlim, ticks=ticks)
    guide_lines(axT, axB, dat['focal'], ypos, labs, dat['series'])
    panel_c(axM, dat['nga'], dat['cmr'], dat['sites'], xlim=(2.3, 16.5),
            ylim=(1.9, 14.2), state_label_xy=dat['state_label_xy'])
    zone_legend(axK, ncol=2)
    tip_key(axK2, dat['tip_key'])

    letter(fig, 0.004, 0.996, 'a')
    letter(fig, 0.004, 0.288, 'b')
    return fig, dict(rows=rows, ypos=ypos, summary=summary,
                     axes=dict(tree=axT, labels=axL, dens=axB, map=axM,
                               zone_key=axK, tip_key=axK2))


def tip_key(ax, entries):
    """Second legend: what the tip and site colours mean."""
    handles = [Line2D([], [], marker='o', linestyle='None', markersize=3.0,
                      markerfacecolor=c, markeredgecolor='0.2',
                      markeredgewidth=0.35, label=lab) for lab, c in entries]
    handles.append(Line2D([], [], marker='o', linestyle='None', markersize=3.0,
                          markerfacecolor='white', markeredgecolor='0.35',
                          markeredgewidth=0.9, label='sampling site'))
    ax.axis('off')
    leg = ax.legend(handles=handles, loc='upper left', fontsize=FS_TICK, ncol=2,
                    frameon=False, handlelength=1.0, columnspacing=0.8,
                    handletextpad=0.35, labelspacing=0.42,
                    title='Genome origin', title_fontsize=FS_TICK,
                    alignment='left')
    return leg


def layout_report(fig, clip=1.0):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    keep, tick_ids = [], set()
    for ax in fig.axes:
        if not ax.axison:                 # axis('off'): its ticks never render
            for axis in (ax.xaxis, ax.yaxis):
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    tick_ids.update({id(tick.label1), id(tick.label2)})
            continue
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                for lab in (tick.label1, tick.label2):
                    tick_ids.add(id(lab))
                    if lab.get_visible() and lab.get_text().strip() and lo <= loc <= hi:
                        keep.append(lab)
    for t in fig.findobj(mpl.text.Text):
        if id(t) not in tick_ids and t.get_visible() and t.get_text().strip():
            keep.append(t)
    T = [(t, t.get_window_extent(rend)) for t in keep]
    ov = [(a.get_text()[:26], b.get_text()[:26]) for i, (a, ba) in enumerate(T)
          for b, bb in T[i + 1:] if ba.overlaps(bb)]
    out = [t.get_text()[:26] for t, bb in T
           if not (bb.x0 >= -clip and bb.y0 >= -clip
                   and bb.x1 <= fig.bbox.x1 + clip and bb.y1 <= fig.bbox.y1 + clip)]
    return {'n_text': len(T), 'overlaps': ov, 'out_of_canvas': out}


# ============================================================== script entry
TREE_FILE = 'tree.nexus'                 # == ZOO+HUM.mcc.tree (identical)
LOG_FILE = 'Complete-NCDC+Zoo+HUM+2026-09.log'
TREES_FILE = 'Complete-NCDC+Zoo+HUM+2026-09.trees'
META_FILE = 'MPXV_Nigeria_curated_updated_with_polygon_coords.csv'
CACHE = 'fig3_posterior_nodes.csv'       # per-tree node ages; reused if present
BURNIN = 0.10
NOT_IN_TREE = ('KJ642615', 'VSP191')     # zoonotic metadata rows with no tip
CMR_REGION = {'north-west': 'Nord-Ouest', 'north west': 'Nord-Ouest',
              'south-west': 'Sud-Ouest', 'south west': 'Sud-Ouest',
              'centre': 'Centre'}
STATE_LABEL_XY = {'Ondo': (3.25, 3.60), 'Edo': (4.35, 2.60), 'Imo': (5.60, 3.60),
                  'Abia': (6.75, 2.60), 'AkwaIbom': (8.15, 3.60),
                  'CrossRiver': (9.25, 2.60)}
TIP_KEY = [('Abia', STATE_C['Abia']), ('Imo', STATE_C['Imo']),
           ('Ondo', STATE_C['Ondo']), ('Edo', STATE_C['Edo']),
           ('Akwa Ibom', STATE_C['AkwaIbom']), ('Cross River', STATE_C['CrossRiver']),
           ('Cameroon', CMR_SAMPLED)]


def _find(root, name):
    p = os.path.join(root, name)
    if os.path.exists(p):
        return p
    for dirpath, _d, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f'{name} not found under {root}')


def sample_key(s):
    s = str(s)
    for pat in (r'(MPOX-\d{2}-\d+)', r'(MPX-\d{2}-\d+)', r'(VSP\d+)', r'(TRM\d+)',
                r'(UNTH-\d+)', r'(CPHL-[A-Za-z0-9]+)', r'\b([A-Z]{2}\d{6})\b'):
        m = re.search(pat, s, re.I)
        if m:
            return m.group(1).upper()
    return s.upper()


def read_tree(path):
    """MCC tree with BEAST annotations; numeric tip labels translated."""
    trans, trees = m4.read_nexus(path)
    root, nodes = m4.parse_newick(trees[0][1])
    for nd in nodes:
        if nd.name in trans:
            nd.name = trans[nd.name]
    return root, nodes


def load_all(data_dir, burnin=BURNIN, tree_file=TREE_FILE, pruned=False):
    """Read every input the figure needs. `pruned=True` draws the tree with the
    1971 Abia genome removed, whose root is the border/south-east divergence."""
    d = os.path.abspath(data_dir)
    root, nodes = read_tree(_find(d, tree_file))
    tips = [x.name for x in nodes if not x.children]
    tip_dates = {t: m4.date_from_label(t) for t in tips}
    tip_dates = {k: v for k, v in tip_dates.items() if v is not None}
    day_only = {t for t in tips if re.search(r'\d{4}-\d{2}-\d{2}', t)}
    C, _ = m4.calibrate(root, tip_dates, day_only=day_only)
    m4.apply_dates(root, C)

    meta = pd.read_csv(_find(d, META_FILE), encoding='latin-1')
    meta = meta.loc[:, ~meta.columns.str.startswith('Unnamed')]
    zc = [c for c in meta.columns if 'animal' in c.lower() or 'h2h' in c.lower()][0]
    sc = [c for c in meta.columns if c.lower() in ('sample', 'seqname', 'id')][0]
    zoo_meta = meta[meta[zc].astype(str).str.strip().str.lower().eq('zoonotic')].copy()
    zkeys = {sample_key(s) for s in zoo_meta[sc]}
    leaf = {x.name: x for x in nodes if not x.children}
    zoo = [x for n, x in leaf.items()
           if sample_key(n) in zkeys or 'cameroon' in n.lower()]
    keep = {x.name for x in zoo}

    # the human epidemic: the largest clade with no zoonotic genome in it
    def n_zoo(nd):
        return sum(1 for l in m4.leaves(nd) if l.name in keep)
    cands = [x for x in nodes if x.children and n_zoo(x) == 0
             and len(m4.leaves(x)) >= 50]
    human = max(cands, key=lambda x: len(m4.leaves(x)))

    def sub(pat):
        return frozenset(x.name for x in zoo if re.search(pat, x.name, re.I))
    border = m4.mrca_fast(root, sub(r'Akwa-?Ibom|CrossRiver|MPOX-24-545|Cameroon'))[0]
    se_human = m4.mrca_fast(root, sub(r'VSP189|MPX-23-046|MPOX-24-1029|MPOX-24-934'
                                      r'|CPHL-572|CPHL-MPOX-25-001'))[0]
    ak4 = m4.mrca_fast(root, sub(r'VSP199|TRM288|MPOX-24-(618|675)'))[0]
    clades = {
        'ingroup': sorted(l.name for l in m4.leaves(root)),
        'border': sorted(l.name for l in m4.leaves(border)),
        'se_human': sorted(l.name for l in m4.leaves(se_human)),
        'ak4': sorted(l.name for l in m4.leaves(ak4)),
        'ak4_stem': sorted(l.name for l in m4.leaves(ak4.parent)),
        'human': sorted(l.name for l in m4.leaves(human)),
    }

    stem = re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(tree_file)[0]).strip('_')
    cache = os.path.join(os.getcwd(), f'fig3_posterior_nodes_{stem}.csv')
    if os.path.exists(cache):
        PD = pd.read_csv(cache)
        print(f'reusing {os.path.basename(cache)} ({len(PD)} posterior trees)')
    else:
        PD = posterior_ages(_find(d, TREES_FILE), clades, tip_dates, day_only, burnin)
        PD.to_csv(cache, index=False)

    log = pd.read_csv(_find(d, LOG_FILE), sep='\t', comment='#', low_memory=False)
    lg = log.iloc[int(len(log) * burnin):]

    spec = focal_spec(pruned)
    series = {}
    for lab, src, col in spec:
        kind, name = src.split(':', 1)
        series[lab] = (lg[name].values if kind == 'log' else PD[name].values, col)
    node_of = {ROOT_LABEL_FULL: root, ROOT_LABEL_PRUNED: root,
               'Nigeria\u2013Cameroon border lineage': border,
               'South-east Nigeria lineage': se_human,
               'Akwa Ibom diverges from South-West Cameroon': ak4.parent,
               'Human epidemic MRCA': human,
               'Akwa Ibom cluster MRCA': ak4}
    focal = [(lab, node_of[lab], col) for lab, _s, col in spec]

    nga = json.load(open(_find(d, 'gadm41_NGA_1.json')))
    cmr = json.load(open(_find(d, 'gadm41_CMR_1.json')))
    drop = NOT_IN_TREE + (('KJ642617',) if pruned else ())
    sites, swapped = site_points(zoo_meta, sc, nga, cmr, drop=drop)

    brackets = [
        ('Nigeria\u2013Cameroon\nborder lineage', border, '#966919', 0.60),
        ('South-east Nigeria lineage\n(contains the human epidemic)',
         se_human, '#097969', 0.60)]
    return dict(root=root, nodes=nodes, human_node=human, keep=keep, C=C,
                focal=focal, brackets=brackets, series=series, nga=nga, cmr=cmr,
                sites=sites, state_label_xy=STATE_LABEL_XY, tip_key=TIP_KEY,
                posterior=PD, log=lg, clades=clades, pruned=pruned,
                xlim=(XLIM_PRUNED if pruned else XLIM),
                ticks=(YEAR_TICKS_PRUNED if pruned else YEAR_TICKS),
                n_swapped=sum(1 for s in swapped if s is True))


def posterior_ages(trees_path, clades, tip_dates, day_only, burnin=BURNIN):
    """Per-tree MRCA and stem ages for each clade, streamed from the .trees file.

    The tree lines of this BEAST X file carry a per-tree annotation block before
    the '=', so each newick is taken from its first '(' rather than parsed off
    the header.
    """
    trans, _ = m4.read_nexus(trees_path, strip_ann=True, max_trees=1)
    inv = {v: k for k, v in trans.items()}
    num = {k: {inv[n] for n in v if n in inv} for k, v in clades.items()}
    for k, v in num.items():
        if len(v) != len(clades[k]):
            raise ValueError(f'{k}: {len(clades[k]) - len(v)} tips not in translate')
    nd = {inv[n]: v for n, v in tip_dates.items() if n in inv}
    ndy = {inv[n] for n in day_only if n in inv}
    with open(trees_path) as fh:
        lines = [ln for ln in fh if ln.lstrip().startswith('tree ')]
    start = int(len(lines) * burnin)
    rows = []
    for i, ln in enumerate(lines):
        if i < start:
            continue
        rt, _ = m4.parse_newick(ln[ln.index('('):], keep_ann=False)
        Ci, _ = m4.calibrate(rt, nd, day_only=ndy)
        dmax = m4.set_depths(rt)
        rec = {'i': i}
        for lab, want in num.items():
            node, parent = m4.mrca_fast(rt, want)
            rec[lab] = Ci - (dmax - node.depth) if node is not None else np.nan
            rec[lab + '_stem'] = (Ci - (dmax - parent.depth)
                                  if node is not None and parent is not None else np.nan)
            rec[lab + '_mono'] = (node is not None
                                  and len(m4.leaves(node)) == len(want))
        rows.append(rec)
        if len(rows) % 800 == 0:
            print(f'  {len(rows)} trees', flush=True)
    print(f'{len(rows)} of {len(lines)} posterior trees used (burn-in '
          f'{burnin:.0%})')
    return pd.DataFrame(rows)


def site_points(zoo_meta, sample_col, nga, cmr, drop=NOT_IN_TREE):
    """Sampling coordinates, with transposed lat/long pairs corrected."""
    df = zoo_meta[~zoo_meta[sample_col].astype(str)
                  .str.contains('|'.join(drop))].copy()

    def admin(r):
        s = str(r[sample_col])
        if 'cameroon' in s.lower():
            m = re.search(r'\|(North-?west|South-?west|Centre[^|]*)\|', s, re.I)
            if m:
                k = m.group(1).lower().replace('centre-cameroon', 'centre')
                return CMR_REGION.get(k, 'Centre')
        return r['State']

    df['admin'] = df.apply(admin, axis=1)
    pts, swapped = fix_coords(df, nga['features'] + cmr['features'], 'admin')

    def colour(r):
        if 'cameroon' in str(r[sample_col]).lower():
            return CMR_SAMPLED
        return STATE_C.get(str(r['State']).replace(' ', ''), '0.4')

    return [(p, colour(r)) for p, (_, r) in zip(pts, df.iterrows())], swapped


PRUNED_TREE_FILE = 'ZOO+HUM.mcc.pruned.tree'


def render(data_dir, out_stem, tree_file=TREE_FILE, pruned=False):
    dat = load_all(data_dir, tree_file=tree_file, pruned=pruned)
    fig, meta = build(dat)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{out_stem}.{ext}', **({'dpi': 400} if ext == 'png' else {}))
    meta['summary'].to_csv(f'{out_stem}_node_ages.csv', index=False)
    rep = layout_report(fig)
    print(f'wrote {out_stem}.png/.pdf/.svg and {out_stem}_node_ages.csv')
    print(f"  tree {tree_file}: {len(dat['keep'])} zoonotic tips, human clade "
          f"{len(m4.leaves(dat['human_node']))}, root "
          f"{dat['root'].x:.2f} | coordinates corrected {dat['n_swapped']}")
    print(f"  layout: {rep['n_text']} text objects, "
          f"{len(rep['out_of_canvas'])} off-canvas, {len(rep['overlaps'])} overlaps")
    return fig, meta


def main(data_dir=None, out_stem='Figure3_zoonotic'):
    """Both figures: the full tree and the pruned tree (1971 Abia removed)."""
    d = data_dir or _HERE
    fig, _m = render(d, out_stem, TREE_FILE, pruned=False)
    figp, _mp = render(d, out_stem + '_pruned', PRUNED_TREE_FILE, pruned=True)
    print(f'both figures written from {d}')
    return fig, figp


if __name__ == '__main__':
    mpl.use('Agg')
    main(*sys.argv[1:3])
