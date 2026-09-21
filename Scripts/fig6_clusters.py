"""Extended Data figure: MPXV cluster size, persistence and dispersal velocity.

Kernel-density panels in the style of Dudas et al. (2017) EBOV phylogeography
(EBOV_phylogeography_kernels.ipynb): a Gaussian KDE on a fixed grid, split at
the normalised 50th and 95th percentiles of its own integral, each segment
drawn in a progressively darker tone with a marker and dropline at its end.

Rows are statistics, columns are groupings (whole epidemic | by wave), as in
the EBOV figure. Cluster statistics come from the per-introduction subtree
table; velocities from the seraphim tree extractions.

Usage:  python fig6_clusters.py [data_dir] [out_stem]
"""
import os
import re
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

FS_BASE, FS_ANN, FS_TICK = 9, 8, 7

# wave windows as used in the main text; the placeholder values shipped in
# mpxv_cluster_supp_waves.py leave 14% of introductions unassigned
WAVES = {'Wave 1': (2015.90, 2021.00), 'Wave 2': (2021.00, 2024.00),
         'Wave 3': (2024.00, 2025.80)}
WAVE_BREAKS = (2016.0, 2021.0, 2024.0, 2026.0)     # branch clipping, as in the R script
KDE_BW = 0.45
GRIDS = {'size': (0.0, 16.0), 'pers': (0.0, 400.0),
         'weighted': (120.0, 380.0), 'mean_branch': (200.0, 1000.0)}
ROWS = [('size', 'cluster size\n(sequences, singletons excluded)', '{:.1f}'),
        ('pers', 'cluster persistence\n(days, singletons excluded)', '{:.0f}'),
        ('weighted', 'weighted lineage dispersal\nvelocity (km per year)', '{:.0f}'),
        ('mean_branch', 'mean branch dispersal\nvelocity (km per year)', '{:.0f}')]


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


def hpd(values, level=0.95):
    v = np.sort(np.asarray(values, float))
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (v[0], v[0]) if len(v) else (np.nan, np.nan)
    k = int(np.floor(level * len(v)))
    m = len(v) - k
    if k < 1 or m <= 0:
        return v[0], v[-1]
    i = int(np.argmin(v[k:] - v[:m]))
    return v[i], v[i + k]


def kde_segments(values, grid, bw=KDE_BW, split='empirical'):
    """KDE split into three segments at the 50th and 95th percentiles.

    split='empirical' cuts at the empirical median and 95th percentile of the
    data, so the markers agree with the quoted summary statistics.
    split='integral' reproduces the EBOV notebook exactly, cutting where the
    KDE's own renormalised integral reaches 0.5 and 0.95; on a strongly skewed
    variable a wide bandwidth puts that well above the empirical median (for
    MPXV cluster persistence, 65 days against a median of 21).
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or np.std(v) == 0:
        return None
    kde = gaussian_kde(v, bw_method=bw)
    y = kde.evaluate(grid)
    lo, hi = float(grid[0]), float(grid[-1])
    if split == 'integral':
        asym = kde.integrate_box_1d(lo, hi)
        if asym <= 0:
            return None
        integral = np.array([kde.integrate_box_1d(lo, x) / asym for x in grid])
        i50 = max(int(np.argmin(np.abs(integral - 0.50))), 1)
        i95 = max(int(np.argmin(np.abs(integral - 0.95))), i50 + 1)
    else:
        q50, q95 = float(np.median(v)), float(np.percentile(v, 95))
        i50 = max(int(np.argmin(np.abs(grid - q50))), 1)
        i95 = max(int(np.argmin(np.abs(grid - q95))), i50 + 1)
    return [(grid[:i50 + 1], y[:i50 + 1]), (grid[i50:i95 + 1], y[i50:i95 + 1]),
            (grid[i95:], y[i95:])]


def tone(colour, k):
    """Progressively darker tones of one colour, for the three KDE segments."""
    rgb = np.array(mcolors.to_rgb(colour))
    return tuple(np.clip(rgb * (1.0 - 0.20 * k), 0, 1))


def draw_kernel(ax, values, grid, colour, *, lw=1.1, ms=13):
    segs = kde_segments(values, grid)
    if segs is None:
        return 0.0
    peak = 0.0
    for k, (xs, ys) in enumerate(segs):
        if len(xs) == 0:
            continue
        c = tone(colour, k)
        ax.plot(xs, ys, color=c, lw=lw, zorder=3 + k, solid_capstyle='round')
        ax.plot([xs[-1], xs[-1]], [0, ys[-1]], color=c, lw=0.7, zorder=3 + k)
        ax.scatter([xs[-1]], [ys[-1]], s=ms, facecolor=c, edgecolor='white',
                   linewidth=0.5, zorder=10 + k)
        peak = max(peak, float(np.nanmax(ys)))
    return peak


def panel(ax, data, grid, colours, *, fmt, ylabel=None, annotate='right',
          singleton_pct=None, note=None):
    """One KDE panel. `data` maps group label -> values; one kernel per group."""
    peak = 0.0
    for lab, vals in data.items():
        peak = max(peak, draw_kernel(ax, vals, grid, colours[lab]))
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, peak * 1.30 if peak > 0 else 1)
    ax.set_yticks([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_ANN, labelpad=3)
    ax.tick_params(axis='x', labelsize=FS_TICK, pad=1.5)
    ax.spines['left'].set_visible(False)

    lines = []
    for lab, vals in data.items():   # markers are the empirical median / p95
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        lo, hi = hpd(v)
        lines.append((lab, f'{fmt.format(np.median(v))} '
                           f'({fmt.format(lo)}\u2013{fmt.format(hi)})', len(v)))
    if annotate == 'right' and len(lines) == 1:
        lab, txt, n = lines[0]
        ax.set_title(f'median {txt},  n = {n:,}', fontsize=FS_TICK, loc='right', pad=2)
    else:
        for i, (lab, txt, n) in enumerate(lines):
            ax.text(0.975, 0.93 - 0.155 * i, f'{lab}  {txt}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=FS_TICK, color=tone(colours[lab], 1))
    if singleton_pct is not None:
        ax.text(0.975, 0.30, f'{singleton_pct:.0f}% of introductions\nare singletons',
                transform=ax.transAxes, ha='right', va='top', fontsize=FS_TICK - 0.5,
                color='0.45', linespacing=1.3)
    if note:
        ax.text(0.975, 0.14, note, transform=ax.transAxes, ha='right', va='top',
                fontsize=FS_TICK - 0.5, color='0.45')
    return peak


def build(data, figsize=(7.2, 8.2)):
    apply_style()
    fig = plt.figure(figsize=figsize, facecolor='w')
    gs = fig.add_gridspec(len(ROWS), 2, hspace=0.42, wspace=0.10,
                          left=0.115, right=0.985, top=0.945, bottom=0.055)
    grey = '#8C8C8C'
    wcol = data['wave_colours']
    axes = []
    for r, (key, ylab, fmt) in enumerate(ROWS):
        lo, hi = GRIDS[key]
        grid = np.linspace(lo, hi, 300)
        a0 = fig.add_subplot(gs[r, 0])
        a1 = fig.add_subplot(gs[r, 1], sharex=a0)
        sing = data['singleton_pct'] if key in ('size', 'pers') else None
        note = ('persistence is right-censored' if key == 'pers' else None)
        panel(a0, {'whole epidemic': data[key]['whole epidemic']}, grid,
              {'whole epidemic': grey}, fmt=fmt, ylabel=ylab,
              singleton_pct=(sing['whole epidemic'] if key == 'size' else None),
              note=note)
        panel(a1, {w: data[key][w] for w in WAVES}, grid, wcol, fmt=fmt,
              annotate='inside', note=note)
        if r == 0:
            a0.set_title('Whole epidemic', fontsize=FS_BASE, loc='left', pad=12)
            a1.set_title('By wave', fontsize=FS_BASE, loc='left', pad=12)
        letter(a0, 'abcd'[r])
        axes.append([a0, a1])
    return fig, axes


def letter(ax, s, x=-0.135, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS_BASE + 1,
            fontweight='bold', va='bottom', ha='right')


def visible_texts(fig):
    """Text objects that actually render; drops tick labels outside the limits."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out, tick_ids = [], set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                for lab in (tick.label1, tick.label2):
                    tick_ids.add(id(lab))
                    if lab.get_visible() and lab.get_text().strip() and lo <= loc <= hi:
                        out.append(lab)
    for t in fig.findobj(mpl.text.Text):
        if id(t) not in tick_ids and t.get_visible() and t.get_text().strip():
            out.append(t)
    return [(t, t.get_window_extent(rend)) for t in out]


def layout_report(fig, clip=1.0):
    T = visible_texts(fig)
    ov = [(a.get_text()[:26], b.get_text()[:26]) for i, (a, ba) in enumerate(T)
          for b, bb in T[i + 1:] if ba.overlaps(bb)]
    out = [t.get_text()[:26] for t, bb in T
           if not (bb.x0 >= -clip and bb.y0 >= -clip
                   and bb.x1 <= fig.bbox.x1 + clip and bb.y1 <= fig.bbox.y1 + clip)]
    return {'n_text': len(T), 'overlaps': ov, 'out_of_canvas': out}


# ============================================================== script entry
def _find(root, name):
    direct = os.path.join(root, name)
    if os.path.exists(direct):
        return direct
    for dirpath, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f'{name} not found under {root}')


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    return 2 * r * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2
                                     + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


def velocity_by_wave(extraction_dir):
    """Weighted lineage velocity and mean branch velocity, per posterior tree.

    Branches are clipped to each wave window exactly as in
    mpxv_dispersal_statistics_by_wave.R (CLIP_BRANCHES = TRUE), so the weighted
    column reproduces dispersal_statistics_by_wave.csv. Distances here are
    haversine rather than the R script's Vincenty ellipsoid, which shifts every
    value by at most ~0.25%. Mean branch velocity is not in the published
    tables and is computed here.
    """
    files = sorted(glob.glob(os.path.join(extraction_dir, 'TreeExtractions_*.csv')),
                   key=lambda p: int(re.search(r'_(\d+)\.csv$', p).group(1)))
    if not files:
        raise FileNotFoundError(f'no TreeExtractions_*.csv in {extraction_dir}')
    rows = []
    for j, f in enumerate(files):
        tb = pd.read_csv(f, usecols=['startLat', 'startLon', 'endLat', 'endLon',
                                     'startYear', 'endYear'])
        sT, eT = tb.startYear.values, tb.endYear.values
        sl, sa, el, ea = (tb.startLon.values, tb.startLat.values,
                          tb.endLon.values, tb.endLat.values)
        dur = eT - sT
        d = haversine_km(sl, sa, el, ea)
        ok = dur > 0
        rows.append(dict(tree=j + 1, wave='whole epidemic',
                         weighted=d[ok].sum() / dur[ok].sum(),
                         mean_branch=float(np.mean(d[ok] / dur[ok])),
                         nBranches=int(ok.sum())))
        for w in range(len(WAVE_BREAKS) - 1):
            t0, t1 = WAVE_BREAKS[w], WAVE_BREAKS[w + 1]
            m = (eT > t0) & (sT < t1) & (dur > 0)
            a = np.maximum(sT[m], t0)
            b = np.minimum(eT[m], t1)
            fa, fb = (a - sT[m]) / dur[m], (b - sT[m]) / dur[m]
            dc = haversine_km(sl[m] + fa * (el[m] - sl[m]), sa[m] + fa * (ea[m] - sa[m]),
                              sl[m] + fb * (el[m] - sl[m]), sa[m] + fb * (ea[m] - sa[m]))
            tc = b - a
            k = tc > 0
            rows.append(dict(tree=j + 1, wave=f'Wave {w + 1}',
                             weighted=dc[k].sum() / tc[k].sum(),
                             mean_branch=float(np.mean(dc[k] / tc[k])),
                             nBranches=int(k.sum())))
    return pd.DataFrame(rows)


def load_all(data_dir):
    d = os.path.abspath(data_dir)
    sub = pd.read_csv(_find(d, 'state+subtrees.tsv'), sep='\t',
                      usecols=['tree_state', 'destination_state', 'introduction_time',
                               'cluster_size', 'persistence_days', 'days_to_mrsd'],
                      dtype={'tree_state': 'int32', 'destination_state': 'category',
                             'cluster_size': 'int32', 'persistence_days': 'float32',
                             'introduction_time': 'float64', 'days_to_mrsd': 'float32'})
    wave = pd.Series(pd.NA, index=sub.index, dtype='object')
    for nm, (a, b) in WAVES.items():
        wave[(sub.introduction_time >= a) & (sub.introduction_time < b)] = nm
    sub['wave'] = wave

    V = velocity_by_wave(os.path.dirname(_find(d, 'TreeExtractions_1.csv')))
    branches = pd.read_csv(_find(d, 'NCDC_MPOX_phylogeography_branches.csv'))
    allv = pd.concat([branches.start_dec_date, branches.end_dec_date])
    norm = mcolors.Normalize(vmin=float(allv.min()), vmax=float(allv.max()))
    wave_colours = {w: cm.plasma(norm(0.5 * (a + b)))
                    for w, (a, b) in WAVES.items()}

    ns = sub[sub.cluster_size > 1]
    groups = {'whole epidemic': ns, **{w: ns[ns.wave == w] for w in WAVES}}
    out = {'size': {}, 'pers': {}, 'weighted': {}, 'mean_branch': {},
           'singleton_pct': {'whole epidemic': float(100 * (sub.cluster_size <= 1).mean())},
           'wave_colours': wave_colours, 'subtrees': sub, 'velocity': V}
    for g, gs_ in groups.items():
        out['size'][g] = gs_.cluster_size.values.astype(float)
        out['pers'][g] = gs_.persistence_days.values.astype(float)
    for w in WAVES:
        s = sub[sub.wave == w]
        out['singleton_pct'][w] = float(100 * (s.cluster_size <= 1).mean())
    for g in ['whole epidemic'] + list(WAVES):
        out['weighted'][g] = V.loc[V.wave == g, 'weighted'].values
        out['mean_branch'][g] = V.loc[V.wave == g, 'mean_branch'].values
    return out


def summarise(data):
    rows = []
    for key, lab in [('size', 'cluster size'), ('pers', 'cluster persistence'),
                     ('weighted', 'weighted lineage dispersal velocity'),
                     ('mean_branch', 'mean branch dispersal velocity')]:
        for g, v in data[key].items():
            v = np.asarray(v, float)
            lo, hi = hpd(v)
            rows.append(dict(panel=lab, group=g, n=len(v), median=np.median(v),
                             p95=np.percentile(v, 95), mean=v.mean(),
                             hpd_lo=lo, hpd_hi=hi,
                             singleton_pct=(data['singleton_pct'].get(g)
                                            if key in ('size', 'pers') else np.nan)))
    V = data['velocity']
    pp = []
    for key, lab in [('weighted', 'weighted lineage dispersal velocity'),
                     ('mean_branch', 'mean branch dispersal velocity')]:
        P = V.pivot(index='tree', columns='wave', values=key)
        for a, b in (('Wave 1', 'Wave 2'), ('Wave 2', 'Wave 3'), ('Wave 1', 'Wave 3')):
            pp.append(dict(statistic=lab, comparison=f'{b} > {a}',
                           posterior_prob=float((P[b] > P[a]).mean())))
    return pd.DataFrame(rows), pd.DataFrame(pp)


def main(data_dir=None, out_stem='ExtendedData_clusters_velocity'):
    d = data_dir or _HERE
    data = load_all(d)
    fig, _axes = build(data)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{out_stem}.{ext}', **({'dpi': 400} if ext == 'png' else {}))
    S, PP = summarise(data)
    S.to_csv('cluster_velocity_summary.csv', index=False)
    PP.to_csv('velocity_posterior_probabilities.csv', index=False)
    data['velocity'].to_csv('velocity_statistics_by_wave.csv', index=False)
    rep = layout_report(fig)
    print(f'wrote {out_stem}.png/.pdf/.svg and three CSVs from {d}')
    print(f"introductions {len(data['subtrees']):,} | "
          f"non-singleton clusters {len(data['size']['whole epidemic']):,} | "
          f"singletons {data['singleton_pct']['whole epidemic']:.1f}%")
    print(f"layout: {rep['n_text']} text objects, {len(rep['out_of_canvas'])} off-canvas, "
          f"{len(rep['overlaps'])} overlaps")
    return fig


if __name__ == '__main__':
    mpl.use('Agg')
    main(*sys.argv[1:3])
