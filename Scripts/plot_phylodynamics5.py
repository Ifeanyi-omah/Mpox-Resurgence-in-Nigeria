#!/usr/bin/env python3
"""
Nigerian MPXV phylodynamics figure.

  a  time-scaled MCC tree, branches coloured by inferred geopolitical zone
  b  SkyGrid effective population size
  c  birth-death skyline R_e by epoch

Vertical markers show when the Sierra Leonean and Togolese lineages diverged
from sampled Nigerian A.2.2 diversity, drawn across all three panels so the
divergences can be read against the resurgence signal.

Usage
-----
  python plot_phylodynamics.py \
      --tree NCDC_2026_state_HPSTR.tree \
      --skygrid SKD.txt \
      -o figure5.pdf
"""

import argparse
import datetime as dt
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42,
                            'ps.fonttype': 42})
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ config
MRSD = 2025.786301369863          # most recent sampling date

ZONE_COLOURS = {'SW': '#6BAED6', 'SS': '#E6772E', 'SE': '#2E8B57',
                'NC': '#8B7BA8', 'NW': '#D9544D', 'NE': '#E8D44D',
                '?': '#CCCCCC'}
ZONE_LABEL = {'SW': 'South West', 'SS': 'South South', 'SE': 'South East',
              'NC': 'North Central', 'NW': 'North West', 'NE': 'North East'}
ZONE_ORDER = ['SW', 'SS', 'SE', 'NC', 'NW', 'NE']

STATE_TO_ZONE = {
    'RIVERS': 'SS', 'BAYELSA': 'SS', 'DELTA': 'SS', 'EDO': 'SS',
    'AKWAIBOM': 'SS', 'CROSSRIVER': 'SS',
    'ABIA': 'SE', 'IMO': 'SE', 'ANAMBRA': 'SE', 'ENUGU': 'SE', 'EBONYI': 'SE',
    'LAGOS': 'SW', 'OGUN': 'SW', 'ONDO': 'SW', 'OSUN': 'SW', 'OYO': 'SW',
    'EKITI': 'SW',
    'KADUNA': 'NW', 'KANO': 'NW', 'SOKOTO': 'NW', 'KEBBI': 'NW',
    'ZAMFARA': 'NW', 'KATSINA': 'NW', 'JIGAWA': 'NW',
    'BENUE': 'NC', 'PLATEAU': 'NC', 'FCT': 'NC', 'NASARAWA': 'NC',
    'NIGER': 'NC', 'KOGI': 'NC', 'KWARA': 'NC',
    'GOMBE': 'NE', 'BORNO': 'NE', 'TARABA': 'NE', 'ADAMAWA': 'NE',
    'BAUCHI': 'NE', 'YOBE': 'NE',
}

LINEAGE_COLOURS = {
    'A': '#C7C7C7', 'A.1': '#9E9E9E', 'A.2': '#56B4E9', 'A.2.1': '#CC79A7',
    'A.2.2': '#404040', 'A.2.3': '#0072B2', 'A.2.4': '#D55E00',
    'A.2.5': '#F0E442', 'A.3': '#8B4513', 'zoonotic': '#D73027',
}

# divergence of each exported lineage from sampled Nigerian A.2.2 diversity
EVENTS = [
    {'label': 'Sierra Leone (G.1)', 'colour': '#B8860B',
     'divergence': '2024-01-23', 'hpd': ('2023-11-13', '2024-04-16')},
    {'label': 'Togo (G.2)', 'colour': '#7B2D8B',
     'divergence': '2024-01-15', 'hpd': ('2023-10-28', '2024-10-14')},
]

# Wave palette shared with Figure 6: the plasma colour map sampled at each
# wave midpoint under Figure 6's branch-date normalisation (2016-03 to
# 2025-10), so a colour denotes the same wave in both figures.
WAVE_COLOURS = {'early': '#7801a8', 'middle': '#eb7655', 'recent': '#fcd225'}

# SkyGrid shading uses Figure 6's epidemic waves, which are NOT the
# birth-death epochs below -- the two analyses cut time differently and the
# panels are labelled accordingly.
SKYGRID_WAVES = [
    {'label': 'Wave 1, 2016\u20132020', 'span': (2015.90, 2021.00),
     'colour': WAVE_COLOURS['early']},
    {'label': 'Wave 2, 2021\u20132023', 'span': (2021.00, 2024.00),
     'colour': WAVE_COLOURS['middle']},
    {'label': 'Wave 3, 2024\u20132025', 'span': (2024.00, 2025.80),
     'colour': WAVE_COLOURS['recent']},
]

# Episodic birth-death epochs, oldest first, in the same palette as the
# SkyGrid waves above. R_e medians and 95% HPDs as reported for this analysis.
EPOCHS = [
    {'label': 'Pre-2022\n2016\u2013Apr 2022', 'colour': WAVE_COLOURS['early'],
     'median': 1.01, 'hpd': (0.99, 1.03)},
    {'label': '2022\u20132024\nMay 2022\u2013Jun 2024', 'colour': WAVE_COLOURS['middle'],
     'median': 0.98, 'hpd': (0.95, 1.02)},
    {'label': '2024\u20132025\nJul 2024\u2013Oct 2025', 'colour': WAVE_COLOURS['recent'],
     'median': 1.13, 'hpd': (1.05, 1.22)},
]


def dec(date_str):
    y, m, d = map(int, date_str.split('-'))
    a = dt.date(y, 1, 1).toordinal()
    n = dt.date(y + 1, 1, 1).toordinal() - a
    return y + (dt.date(y, m, d).toordinal() - a) / n


def zone_of(label):
    parts = [p.strip() for p in label.split('|')]
    for p in parts:
        k = p.upper().replace(' ', '').replace('-', '').replace('_', '')
        if k in STATE_TO_ZONE:
            return STATE_TO_ZONE[k]
    return '?'


# ------------------------------------------------------------ tree parsing
def read_tree(path):
    """Return children, brlen, annotations, labels for a BEAST summary tree."""
    raw = open(path).read()

    labels = {}
    i = raw.find('Translate')
    if i != -1:
        for line in raw[i:raw.find('tree', i)].split('\n')[1:]:
            line = line.strip().rstrip(',').rstrip(';')
            p = line.split(None, 1)
            if len(p) == 2 and p[0].isdigit():
                labels[int(p[0])] = p[1].strip().strip("'")
    else:
        i0, j0 = raw.find('Taxlabels'), raw.find('Begin trees')
        names = [l.strip().strip("'") for l in raw[i0:j0].split('\n')[1:]
                 if l.strip() and l.strip() not in (';', 'End;')]
        labels = {k + 1: n for k, n in enumerate(names)}
    name_to_id = {v: k for k, v in labels.items()}

    nw = [l for l in raw.split('\n') if l.lstrip().startswith('tree ')][0]
    nw = nw[nw.index('=') + 1:].strip()
    if nw.startswith('[&R]'):
        nw = nw[4:].strip()

    children, brlen, annot = {}, {}, {}
    stack, counter, cur = [], [0], None
    i, n = 0, len(nw)
    while i < n:
        c = nw[i]
        if c == '(':
            counter[0] -= 1
            node = counter[0]
            children[node] = []
            if stack:
                children[stack[-1]].append(node)
            stack.append(node)
            cur = None
            i += 1
        elif c == ',':
            i += 1
        elif c == ')':
            cur = stack.pop()
            i += 1
        elif c == '[':
            depth, j = 1, i + 1
            while j < n and depth:
                if nw[j] == '[':
                    depth += 1
                elif nw[j] == ']':
                    depth -= 1
                j += 1
            if cur is not None:
                annot[cur] = nw[i:j]
            i = j
        elif c == ':':
            j = i + 1
            while j < n and (nw[j].isdigit() or nw[j] in '.eE+-'):
                j += 1
            brlen[cur] = float(nw[i + 1:j])
            i = j
        elif c == ';':
            break
        else:
            j = i
            while j < n and nw[j] not in '(),:;[':
                j += 1
            tok = nw[i:j].strip().strip("'")
            if tok:
                cur = int(tok) if tok.isdigit() else name_to_id.get(tok)
                if cur is None:
                    counter[0] -= 1
                    cur = counter[0]
                children.setdefault(cur, [])
                if stack:
                    children[stack[-1]].append(cur)
            i = j
    return children, brlen, annot, labels


def get_trait(annot, node, key):
    s = annot.get(node, '')
    m = re.search(rf'{re.escape(key)}="?([A-Za-z_0-9.]+)"?', s)
    return m.group(1) if m else None


def layout(children, root=-1):
    heights, order, stack = {root: 0.0}, [root], [root]
    while stack:
        p = stack.pop()
        for ch in children.get(p, []):
            stack.append(ch)
            order.append(ch)
    ypos, counter = {}, [0]
    sys.setrecursionlimit(50000)

    def walk(node):
        kids = children.get(node, [])
        if not kids:
            ypos[node] = counter[0]
            counter[0] += 1
            return ypos[node]
        ys = [walk(k) for k in kids]
        ypos[node] = float(np.mean([min(ys), max(ys)]))
        return ypos[node]

    walk(root)
    return ypos, order


# --------------------------------------------------------------- annotation
def mark_events(ax, events=EVENTS, hpd=True, label=False, fontsize=9):
    # labels are staggered because the two divergences are only days apart
    for i, ev in enumerate(events):
        x = dec(ev['divergence'])
        if hpd and ev.get('hpd'):
            lo, hi = (dec(d) for d in ev['hpd'])
            ax.axvspan(lo, hi, facecolor=ev['colour'], alpha=0.10, lw=0, zorder=0)
        ax.axvline(x, color=ev['colour'], lw=1.6, ls='--', alpha=0.9, zorder=6)
        if label:
            span = ax.get_xlim()[1] - ax.get_xlim()[0]
            ax.annotate(ev['label'],
                        xy=(x, 1.0), xycoords=ax.get_xaxis_transform(),
                        xytext=(x + (0.11 + 0.11 * i) * span, 1.06 - 0.055 * i),
                        textcoords=ax.get_xaxis_transform(),
                        ha='left', va='center', color=ev['colour'],
                        fontsize=fontsize, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=ev['colour'],
                                        lw=1.1, shrinkA=0, shrinkB=2,
                                        connectionstyle='angle,angleA=0,angleB=90,rad=3'),
                        annotation_clip=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', required=True)
    p.add_argument('--skygrid', required=True)
    p.add_argument('-o', '--out', default='phylodynamics.pdf')
    p.add_argument('--formats', default='png,svg')
    p.add_argument('--mrsd', type=float, default=MRSD)
    p.add_argument('--xmin', type=float, default=None,
                   help='default: the first SkyGrid time point, so the '
                        'trajectory starts flush against the y axis')
    p.add_argument('--xmax', type=float, default=2026.0)
    p.add_argument('--figsize', default='13,15')
    p.add_argument('--font-size', type=float, default=19)
    p.add_argument('--no-events', action='store_true')
    p.add_argument('--nextclade', help='Nextclade TSV; adds lineage bars to panel a')
    p.add_argument('--lineage-min', type=int, default=3,
                   help='minimum tips for a lineage block to be drawn')
    p.add_argument('--lineage-gap', type=int, default=6,
                   help='max vertical gap tolerated within one lineage block')
    p.add_argument('--legend-size', type=float, default=None,
                   help='legend font size (default: --font-size)')
    args = p.parse_args()

    fs = args.font_size

    # ---- SkyGrid (read first: it sets the left edge of the shared time axis)
    sk = np.genfromtxt(args.skygrid, delimiter='\t', names=True,
                       skip_header=1, dtype=None, encoding='utf-8')
    sk_t = np.array([float(x) for x in sk['time']])
    sk_mean = np.array([float(x) for x in sk['mean']])
    sk_lo = np.array([float(x) for x in sk['lower']])
    sk_hi = np.array([float(x) for x in sk['upper']])
    _o = np.argsort(sk_t)
    sk_t, sk_mean, sk_lo, sk_hi = sk_t[_o], sk_mean[_o], sk_lo[_o], sk_hi[_o]
    if args.xmin is None:
        args.xmin = float(sk_t[0])          # no gap at the y axis

    # ---- tree ------------------------------------------------------------
    children, brlen, annot, labels = read_tree(args.tree)
    ypos, order = layout(children)

    depth = {-1: 0.0}
    stack = [-1]
    while stack:
        node = stack.pop()
        for ch in children.get(node, []):
            depth[ch] = depth[node] + brlen.get(ch, 0.0)
            stack.append(ch)
    maxd = max(depth.values())
    times = {k: args.mrsd - (maxd - v) for k, v in depth.items()}

    zone = {}
    for node in depth:
        if node > 0:
            zone[node] = zone_of(labels.get(node, ''))
        else:
            loc = get_trait(annot, node, 'location')
            zone[node] = STATE_TO_ZONE.get(
                (loc or '').upper().replace('_', '').replace('-', ''), '?')

    print(f'{len([k for k in depth if k > 0])} tips; '
          f'root {times[-1]:.2f}', file=sys.stderr)

    # ---- figure ----------------------------------------------------------
    w, h = (float(x) for x in args.figsize.split(','))
    fig, (ax, axs, axr) = plt.subplots(
        3, 1, figsize=(w, h),
        gridspec_kw={'height_ratios': [3.05, 1.20, 1.18], 'hspace': 0.30})

    # -- a: tree
    segs, cols = [], []
    for node in depth:
        if node == -1:
            continue
        parent = next((q for q in children if node in children[q]), None)
        if parent is None:
            continue
        c = ZONE_COLOURS.get(zone[node], '#CCCCCC')
        segs.append([(times[parent], ypos[node]), (times[node], ypos[node])])
        cols.append(c)
    for node in children:
        kids = children[node]
        if node < 0 and kids:
            ys = [ypos[k] for k in kids]
            segs.append([(times[node], min(ys)), (times[node], max(ys))])
            cols.append(ZONE_COLOURS.get(zone[node], '#CCCCCC'))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.1, zorder=2))

    tips = [k for k in depth if k > 0]
    ax.scatter([times[k] for k in tips], [ypos[k] for k in tips],
               s=42, c=[ZONE_COLOURS.get(zone[k], '#CCC') for k in tips],
               edgecolor='k', linewidths=0.5, zorder=5)

    # lineage bars along the right edge of the tree
    lineage = {}
    if args.nextclade and os.path.exists(args.nextclade):
        import csv as _csv
        with open(args.nextclade) as fh:
            nc = {r['seqName'].strip(): (r['lineage'] or 'zoonotic').strip()
                  or 'zoonotic' for r in _csv.DictReader(fh, delimiter='\t')}
        pre = {k.split('|')[0]: v for k, v in nc.items()}
        for k in tips:
            lab = labels.get(k, '')
            lineage[k] = nc.get(lab) or pre.get(lab.split('|')[0])
        n_lin = len({v for v in lineage.values() if v})
        print(f'{sum(1 for v in lineage.values() if v)} tips assigned to '
              f'{n_lin} lineages', file=sys.stderr)

    ax.set_ylim(-4, max(ypos.values()) + 4)
    ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    if lineage:
        by_lin = defaultdict(list)
        for k, v in lineage.items():
            if v:
                by_lin[v].append(ypos[k])
        bar_x = args.xmax + 0.012 * (args.xmax - args.xmin)
        for lin, ys in by_lin.items():
            ys = sorted(ys)
            blocks, cur = [], [ys[0]]
            for y in ys[1:]:
                if y - cur[-1] <= args.lineage_gap:
                    cur.append(y)
                else:
                    blocks.append(cur); cur = [y]
            blocks.append(cur)
            blocks = [b for b in blocks if len(b) >= args.lineage_min]
            if not blocks:
                continue
            col = LINEAGE_COLOURS.get(lin, '#999999')
            biggest = max(blocks, key=len)
            for b in blocks:
                ax.plot([bar_x, bar_x], [min(b) - 0.5, max(b) + 0.5],
                        color=col, lw=5.0, solid_capstyle='butt',
                        zorder=7, clip_on=False)
                if b is biggest:
                    ax.text(bar_x + 0.022 * (args.xmax - args.xmin),
                            (min(b) + max(b)) / 2, lin, color=col,
                            fontsize=fs, fontweight='bold', va='center',
                            ha='left', zorder=8, clip_on=False)

    present = [z for z in ZONE_ORDER if z in set(zone.values())]
    ax.legend(handles=[Line2D([0], [0], marker='o', ls='None', markersize=9,
                              markerfacecolor=ZONE_COLOURS[z],
                              markeredgecolor='k', markeredgewidth=0.5,
                              label=ZONE_LABEL[z]) for z in present],
              loc='center left',
              bbox_to_anchor=(1.10 if lineage else 1.01, 0.5),
              frameon=False, fontsize=args.legend_size or fs)

    # -- b: SkyGrid
    t, mean, lo, hi = sk_t, sk_mean, sk_lo, sk_hi
    # the trajectory is drawn one wave at a time in that wave's colour, with
    # consecutive segments sharing their boundary sample so the line stays
    # joined; the shading behind it marks the same waves
    for w in SKYGRID_WAVES:
        t0, t1 = w['span']
        axs.axvspan(max(t0, args.xmin), min(t1, args.xmax),
                    facecolor=w['colour'], alpha=0.10, lw=0, zorder=0)
        m = (t >= t0) & (t <= t1)
        if m.sum() < 2:
            continue
        j = np.flatnonzero(m)
        j = np.r_[max(j[0] - 1, 0), j, min(j[-1] + 1, len(t) - 1)]
        axs.fill_between(t[j], lo[j], hi[j], color=w['colour'], alpha=0.32,
                         lw=0, zorder=2)
        axs.plot(t[j], mean[j], color=w['colour'], lw=2.6, zorder=3,
                 solid_capstyle='round')
    for w in SKYGRID_WAVES[1:]:
        if args.xmin <= w['span'][0] <= args.xmax:
            axs.axvline(w['span'][0], color='white', lw=1.0, zorder=1)
    axs.set_yscale('log')
    axs.set_ylabel('Effective\npopulation size', fontsize=fs)
    axs.legend(handles=[Line2D([0], [0], color=w['colour'], lw=2.6,
                               label=w['label']) for w in SKYGRID_WAVES],
               frameon=False, fontsize=fs * 0.8, loc='upper left', ncol=3,
               columnspacing=1.0, handlelength=1.4, handletextpad=0.4)
    for s in ('top', 'right'):
        axs.spines[s].set_visible(False)

    # -- c: R_e epochs, oldest at the top as in the original layout
    for i, ep in enumerate(EPOCHS):
        y = len(EPOCHS) - 1 - i
        m, (l, u) = ep['median'], ep['hpd']
        sd = (u - l) / 3.92
        g = np.linspace(l - 2 * sd, u + 2 * sd, 400)
        d = np.exp(-0.5 * ((g - m) / sd) ** 2)
        d = d / d.max() * 0.78
        axr.fill_between(g, y, y + d, color=ep['colour'], alpha=0.65, lw=0)
        axr.plot(g, y + d, color=ep['colour'], lw=1.2)
        axr.plot([m, m], [y, y + 0.78], color='white', lw=1.6)
        axr.text(u + 0.012, y + 0.36,
                 f"median {m:.2f}; 95% HPD {l:.2f}–{u:.2f}",
                 fontsize=fs * 0.85, va='center')
    axr.axvline(1.0, color='k', ls='--', lw=1.2)
    axr.text(1.0, len(EPOCHS) - 0.02, r'  $R_e = 1$', fontsize=fs * 0.9,
             ha='left', va='bottom')
    axr.set_yticks([len(EPOCHS) - 1 - i + 0.35 for i in range(len(EPOCHS))])
    axr.set_yticklabels([e['label'] for e in EPOCHS], fontsize=fs * 0.85)
    axr.set_xlabel(r'Effective reproduction number, $R_e$', fontsize=fs)
    axr.set_ylim(-0.1, len(EPOCHS) + 0.42)
    axr.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        axr.spines[s].set_visible(False)

    # -- shared time axis and event markers
    for a in (ax, axs):
        a.set_xlim(args.xmin, args.xmax)
    for yr in range(int(args.xmin), int(args.xmax) + 1, 2):        # tree only
        ax.axvspan(yr, yr + 1, facecolor='k', alpha=0.04, lw=0, zorder=0)
    ax.set_xticklabels([])
    axs.set_xlabel('Year', fontsize=fs)
    for a in (ax, axs):
        a.tick_params(labelsize=fs * 0.9)

    if not args.no_events:
        mark_events(ax, label=True, fontsize=fs * 0.8)
        mark_events(axs, label=False)

    for a, lab in ((ax, 'a'), (axs, 'b'), (axr, 'c')):
        a.text(-0.055, 1.0, lab, transform=a.transAxes, fontsize=fs * 1.8,
               fontweight='bold', va='bottom', ha='right')

    stem, ext = os.path.splitext(args.out)
    for e in dict.fromkeys([ext.lstrip('.') or 'pdf'] +
                           [x.strip() for x in args.formats.split(',') if x.strip()]):
        path = f'{stem}.{e}'
        kw = {'dpi': 300} if e in ('png', 'jpg', 'tif') else {}
        fig.savefig(path, bbox_inches='tight', facecolor='w', **kw)
        print(f'wrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
