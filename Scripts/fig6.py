"""Figure 6 (MPXV Nigeria spatial spread) drawn at final print size.

Reuses the map and data-summary functions of mpox_nature_composite_v7.py, but
redraws panels a-c at Nature double-column size (183 mm) with the project's
font ladder, implements the to/from fix for the route panels, and adds the
dispersal-statistics panel that replaces the reference map.

Panels:
    a  region-to-region introductions per month, faceted by origin zone
    b  top state routes, 2024-2025            (to/from split, no route arrow)
    c  top region routes, 2024-2025
    d  continuous phylogeography, three waves + shared date scale
    e  weighted dispersal velocity, weighted diffusion coefficient and
       wavefront distance through time, by wave
The reference map of states and zones moves to Extended Data.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _HERE)
import mpox_nature_composite_v7 as v7      # data readers, map drawing, palettes

# ------------------------------------------------------------------ typography
FS_BASE, FS_ANN, FS_TICK = 9.0, 8.0, 7.0
FS_FACET, FS_FACET_TICK = 7.0, 6.5
FS_LETTER = 10.0

WAVE_ORDER = ['Wave 1 (2016-2020)', 'Wave 2 (2021-2023)', 'Wave 3 (2024-2025)']
WAVE_SHORT = {'Wave 1 (2016-2020)': 'Wave 1\n2016-2020',
              'Wave 2 (2021-2023)': 'Wave 2\n2021-2023',
              'Wave 3 (2024-2025)': 'Wave 3\n2024-2025'}
WAVE_C = {'Wave 1 (2016-2020)': '#9ECAE1', 'Wave 2 (2021-2023)': '#3F7FB8',
          'Wave 3 (2024-2025)': '#0B3D66'}


def apply_style():
    """Final-size rcParams: text stays editable in both vector formats."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': FS_BASE, 'axes.labelsize': FS_BASE, 'axes.titlesize': FS_BASE,
        'legend.fontsize': FS_ANN, 'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'legend.frameon': False,
        'figure.dpi': 200, 'savefig.dpi': 400,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'ps.fonttype': 42,
        'lines.linewidth': 0.9, 'patch.linewidth': 0.5,
    })
    # the v7 map helpers read these module constants at call time
    v7.LW_MIN, v7.LW_MAX = 0.18, 1.5
    v7.POINT_SIZE_END = 3.2
    v7.HALO_EXTRA_WIDTH = 0.7
    v7.MAP_TITLE_FS = FS_ANN
    v7.MAP_LABEL_FS = 5.2
    v7.FS_AXIS, v7.FS_TICK, v7.FS_SMALL = FS_BASE, FS_TICK, FS_ANN
    v7.FS_TITLE, v7.FS_LEGEND, v7.FS_PANEL = FS_BASE, FS_ANN, FS_LETTER


def visible_texts(fig):
    """Text objects that actually render: drops tick labels whose tick value
    falls outside the axes view limits, which matplotlib culls at draw time
    but still reports through findobj/get_window_extent."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible() and lab.get_text().strip() and lo <= loc <= hi:
                        out.append(lab)
    tick_ids = {id(t) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                for tick in axis.get_major_ticks() + axis.get_minor_ticks()
                for t in (tick.label1, tick.label2)}
    for t in fig.findobj(mpl.text.Text):
        if id(t) not in tick_ids and t.get_visible() and t.get_text().strip():
            out.append(t)
    return [(t, t.get_window_extent(rend)) for t in out]


def layout_report(fig, clip=1.0):
    """Pairwise text collisions and out-of-canvas text, as a printable summary."""
    T = visible_texts(fig)
    ov = [(a.get_text()[:26], b.get_text()[:26])
          for i, (a, ba) in enumerate(T) for b, bb in T[i + 1:] if ba.overlaps(bb)]
    out = [t.get_text()[:26] for t, bb in T
           if not (bb.x0 >= -clip and bb.y0 >= -clip
                   and bb.x1 <= fig.bbox.x1 + clip and bb.y1 <= fig.bbox.y1 + clip)]
    return {'n_text': len(T), 'overlaps': ov, 'out_of_canvas': out}


def letter(ax, s, x=-0.055, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS_LETTER,
            fontweight='bold', va='bottom', ha='left')


def set_frame(ax, mode='open'):
    for sp, on in [('top', False), ('right', False),
                   ('left', mode != 'none'), ('bottom', mode != 'none')]:
        ax.spines[sp].set_visible(on)


# =================================================================== panel a
def panel_a(fig, subspec, wide, *, full_xlim=(2015.8, 2025.9),
            late_xlim=(2024.0, 2025.9), first_case=2017.71,
            facet_order=('SS', 'SW', 'SE', 'NC'), level='region',
            label_map=None, n_facets=None, late=True, key_title=None,
            dest_level=None):
    """Stacked monthly introductions, one row per origin zone.

    Adapted from v7.draw_region_timing_faceted for final print size: smaller
    key markers, thinner rules, and the zone swatch drawn inside the axes.
    """
    if dest_level == 'region' and level == 'state':
        # destinations collapsed to zones: six colours instead of one per state,
        # and the same zone colours as Figure 6a
        agg = {}
        for p in wide.columns:
            o, d = p.split('___', 1)
            key = f'{o}___{v7.STATE_TO_REGION.get(d, "Unknown")}'
            agg[key] = agg.get(key, 0) + wide[p]
        wide = pd.DataFrame(agg)
        wide = wide[wide.sum(axis=0).sort_values(ascending=False).index]
    palette = v7.get_palette(level)
    labeller = (label_map if label_map is not None else
                (v7.REGION_LABELS if level == 'region' else {}))
    def _lab(code):
        return labeller.get(code, v7.pretty_location_label(code)
                            if level == 'state' else code)
    pair_origin = {p: p.split('___', 1)[0] for p in wide.columns}
    if facet_order is None:                      # busiest origins, by total
        tot = {}
        for p, o in pair_origin.items():
            tot[o] = tot.get(o, 0.0) + float(wide[p].sum())
        facet_order = [o for o, _ in sorted(tot.items(), key=lambda kv: -kv[1])]
        if n_facets:
            facet_order = facet_order[:n_facets]
    origins = [r for r in facet_order if r in set(pair_origin.values())]
    n = len(origins)
    panel_cols, pmax = {}, 0.0
    for r in origins:
        cols = [p for p, o in pair_origin.items() if o == r]
        cols = wide[cols].sum(axis=0).sort_values(ascending=False).index.tolist()
        panel_cols[r] = cols
        pmax = max(pmax, wide[cols].sum(axis=1).max())
    ymax = pmax * 1.18 if pmax > 0 else 1

    ncol = 2 if late else 1
    widths = ([full_xlim[1] - full_xlim[0], late_xlim[1] - late_xlim[0]]
              if late else [1.0])
    inner = subspec.subgridspec(
        n + 1, ncol, hspace=0.30, wspace=0.055,
        height_ratios=[1.0] * n + [0.42], width_ratios=widths)
    tt = wide.index.values
    axes = [[None, None] for _ in range(n)]

    def _draw(ax, cols, xlim):
        cum = np.zeros(len(wide))
        for pair in cols:
            _, end = pair.split('___', 1)
            new = cum + wide[pair].values
            ecol = (v7.get_palette('region').get(end, '#C0C0C0')
                    if dest_level == 'region' else palette.get(end, '#C0C0C0'))
            ax.fill_between(tt, cum, new, color=ecol,
                            alpha=0.85, edgecolor='black', linewidth=0.12)
            cum = new
        ax.set_xlim(*xlim)
        ax.set_ylim(0, ymax)
        set_frame(ax)
        if xlim[0] <= first_case <= xlim[1]:
            ax.axvline(first_case, color='#D62728', ls=(0, (2.2, 1.4)), lw=0.65, zorder=5)
        ax.tick_params(axis='both', labelsize=FS_FACET_TICK, length=2.0, pad=1.4)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(2, integer=True))
        ax.set_xticks([t for t in range(2016, 2026, 2) if xlim[0] <= t <= xlim[1]]
                      if xlim[1] - xlim[0] > 4 else
                      [t for t in (2024, 2025) if xlim[0] <= t <= xlim[1]])

    for row, r in enumerate(origins):
        a0 = fig.add_subplot(inner[row, 0])
        a1 = fig.add_subplot(inner[row, 1], sharey=a0) if late else None
        axes[row] = [a0, a1]
        _draw(a0, panel_cols[r], full_xlim)
        if late:
            _draw(a1, panel_cols[r], late_xlim)
        a0.add_patch(Rectangle((0.010, 0.70), 0.020, 0.24, transform=a0.transAxes,
                               facecolor=palette.get(r, '#C0C0C0'), edgecolor='black',
                               linewidth=0.35, clip_on=False, zorder=20))
        a0.text(0.040, 0.82, f'From {_lab(r)}', transform=a0.transAxes,
                fontsize=FS_FACET, fontweight='bold', va='center', ha='left', zorder=21)
        if late:
            a1.tick_params(axis='y', labelleft=False)
        if row < n - 1:
            a0.tick_params(labelbottom=False)
            if late:
                a1.tick_params(labelbottom=False)

    if late:
        axes[0][0].set_title('Full epidemic', fontsize=FS_ANN, pad=3)
        axes[0][1].set_title('Late phase', fontsize=FS_ANN, pad=3)
    for j in range(ncol):
        axes[-1][j].set_xlabel('Year', fontsize=FS_ANN, labelpad=0.8)
    axes[n // 2][0].set_ylabel('Mean introductions per month', fontsize=FS_ANN,
                               labelpad=2.5, y=0.0)

    dest = sorted({p.split('___', 1)[1] for p in wide.columns},
                  key=lambda r: v7.REGION_RANK.get(r, 999) if level == 'region'
                  else str(r))
    dpal = v7.get_palette('region') if dest_level == 'region' else palette
    dlab = ((lambda c: v7.REGION_LABELS.get(c, c)) if dest_level == 'region'
            else _lab)
    handles = [mlines.Line2D([], [], marker='s', color='w', markersize=3.6,
                             markerfacecolor=dpal.get(d, '#C0C0C0'),
                             markeredgecolor='black', markeredgewidth=0.3,
                             label=dlab(d)) for d in dest]
    leg = fig.add_subplot(inner[n, :])
    leg.axis('off')
    header = mlines.Line2D([], [], linestyle='None', marker='None',
                           label=key_title or ('Destination state:'
                                               if level == 'state' and dest_level != 'region'
                                               else 'Destination zone:'))
    leg.legend(handles=[header] + handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.58), frameon=False, fontsize=FS_TICK,
               ncol=min(len(handles) + 1, 9), columnspacing=0.85,
               handletextpad=0.25, handlelength=1.0)
    return axes


# ============================================================= panels b and c
RIBBON_HALF = 0.34


def _darken(hexc, f=0.72):
    c = np.array(mpl.colors.to_rgb(hexc)) * f
    return mpl.colors.to_hex(np.clip(c, 0, 1))


def _density(values, max_width=RIBBON_HALF, mass=0.95):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([]), np.nan
    med = float(np.median(v))
    lo, hi = v7.hpd_interval(v, mass)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = hi = med
    if hi < lo:
        lo, hi = hi, lo
    if np.unique(v).size < 2 or np.isclose(lo, hi):
        return np.array([]), np.array([]), med    # point mass: median tick only
    xg = np.linspace(lo, hi, 200)
    try:
        d = gaussian_kde(v).evaluate(xg)
        d = d / np.nanmax(d) * max_width if np.nanmax(d) > 0 else np.full_like(xg, 0.08)
    except Exception:
        d = np.full_like(xg, 0.08)
    return xg, d, med


def panel_routes(ax, route_summary, from_summary, to_summary, *, level,
                 period_label, max_x, top_n=10, label_gap=0.19, show_key=True,
                 label_fs=6.0):
    """Top directed routes as split densities.

    The two halves are marginal totals - everything OUT of the origin (lower)
    and everything INTO the destination (upper) - not the route-specific count,
    so no arrow is drawn between the two place names. Instead each name is
    placed on the side whose density it describes and tinted to match it, and
    a stacked arrow key states the convention once.
    """
    palette = v7.get_palette(level)
    top_routes, _ = v7.select_top_routes(route_summary, top_n=top_n)
    if not top_routes:
        ax.text(0.5, 0.5, f'No {level} routes for {period_label}',
                transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANN)
        return
    sel = route_summary[route_summary['route'].isin(top_routes)].copy()
    ordered = list(reversed(top_routes))
    n = len(ordered)

    for i, route in enumerate(ordered):
        row = sel[sel['route'] == route]
        s_loc, e_loc = row['startLocation'].iloc[0], row['endLocation'].iloc[0]
        s_lab, e_lab = row['start_label'].iloc[0], row['end_label'].iloc[0]
        from_c = palette.get(s_loc, v7.region_color(s_loc))
        to_c = palette.get(e_loc, v7.region_color(e_loc))

        f_vals = from_summary.loc[from_summary['location'] == s_loc, 'count'].to_numpy(float)
        t_vals = to_summary.loc[to_summary['location'] == e_loc, 'count'].to_numpy(float)

        for vals, colour, sign in ((f_vals, from_c, -1), (t_vals, to_c, +1)):
            xg, d, med = _density(vals)
            # a posterior narrower than ~1.5% of the axis cannot be read as a
            # density; draw it as a median tick with an HPD whisker instead
            if not xg.size or (xg.max() - xg.min()) < max_x * 0.015:
                if np.isfinite(med):
                    ax.vlines(med, i, i + sign * RIBBON_HALF * 0.8, color=colour,
                              lw=1.1, zorder=5)
                    if xg.size:
                        ax.hlines(i + sign * RIBBON_HALF * 0.8, xg.min(), xg.max(),
                                  color=colour, lw=0.5, zorder=5)
                continue
            base = np.full_like(xg, i)
            ax.fill_between(xg, base, i + sign * d, facecolor=colour, edgecolor='none',
                            alpha=0.85, zorder=3)
            ax.plot(xg, i + sign * d, color='black', lw=0.3, zorder=4)
            ax.plot(xg, base, color='black', lw=0.2, zorder=4)
            ax.vlines(med, i, i + sign * RIBBON_HALF, color='black', lw=0.6, zorder=5)

        s_disp = s_loc if level == 'region' else s_lab
        e_disp = e_loc if level == 'region' else e_lab
        yt = ax.get_yaxis_transform()
        ax.text(-0.02, i + label_gap, e_disp, transform=yt, ha='right', va='center',
                fontsize=label_fs, fontweight='bold', color=_darken(to_c), clip_on=False)
        ax.text(-0.02, i - label_gap, s_disp, transform=yt, ha='right', va='center',
                fontsize=label_fs, color=_darken(from_c), clip_on=False)

    ax.set_yticks([])
    for b in range(0, n, 2):
        ax.axhspan(b - 0.5, b + 0.5, color='black', alpha=0.030, lw=0, zorder=0)
    ax.set_xlim(0, max_x)
    ax.set_ylim(-0.55, n - 0.45 + (1.15 if show_key else 0.0))
    ax.set_xlabel('Transitions per posterior tree', fontsize=FS_ANN, labelpad=1.5)
    ax.set_title(f"{'State' if level == 'state' else 'Zone'} transitions, {period_label}",
                 fontsize=FS_BASE, loc='left', pad=3)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.6, zorder=0)
    set_frame(ax, 'bottom_only')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='x', labelsize=FS_TICK, length=2.0, pad=1.5)

    if show_key:
        # Pekar: no single directional arrow between the two place names. A
        # right arrow above a left arrow, drawn clear of the data above the
        # axes, states which half is "to" and which is "from".
        # drawn in the reserved headroom band above the top route, so it moves
        # with the data rather than with the figure margins
        xa, xb = max_x * 0.30, max_x * 0.37
        for dy, lab, forward in ((0.46, 'upper = into destination', True),
                                 (0.06, 'lower = out of origin', False)):
            yy = n - 0.30 + dy
            x0, x1 = (xa, xb) if forward else (xb, xa)
            ax.annotate('', xy=(x1, yy), xytext=(x0, yy),
                        arrowprops=dict(arrowstyle='-|>', lw=0.55, color='0.3',
                                        mutation_scale=4.5, shrinkA=0, shrinkB=0))
            ax.text(xb + max_x * 0.015, yy, lab, fontsize=6.0,
                    va='center', ha='left', color='0.3')


# =================================================================== panel d
def panel_d(fig, subspec, edge, bg, xlim, ylim, periods=None, cmap=None):
    """Three wave maps sharing one branch-date colour scale."""
    from matplotlib import cm, colors as mcolors
    periods = periods or v7.PHYLO_PERIODS
    cmap = cmap or cm.plasma
    inner = subspec.subgridspec(2, 3, height_ratios=[1, 0.075], wspace=0.04, hspace=0.12)
    allv = pd.concat([edge.start_dec_date, edge.end_dec_date])
    norm = mcolors.Normalize(vmin=float(allv.min()), vmax=float(allv.max()))
    axes, counts = [], {}
    for i, (lab, (a, b)) in enumerate(periods.items()):
        ax = fig.add_subplot(inner[0, i])
        axes.append(ax)
        sub = v7.filter_edges_by_period(edge, a, b)
        counts[lab.replace('\n', ' ')] = len(sub)
        v7.draw_phylo_period_panel(ax, sub, bg, norm, cmap,
                                   f'{lab}  (n = {len(sub)})', xlim=xlim, ylim=ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{lab}  (n = {len(sub)})', fontsize=FS_ANN, pad=2)
    cax = fig.add_subplot(inner[1, :])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    ticks = np.linspace(norm.vmin, norm.vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([v7.convertDate(v7.decimal_to_date(t), '%Y-%m-%d', '%Y-%m')
                       for t in ticks])
    cb.set_label('Branch date', fontsize=FS_ANN, labelpad=1.2)
    cb.ax.tick_params(labelsize=FS_TICK, width=0.5, length=2, pad=1.0)
    cb.outline.set_linewidth(0.4)
    # inset the bar so the first and last tick labels stay on the canvas
    p = cax.get_position()
    cax.set_position([p.x0 + p.width * 0.045, p.y0, p.width * 0.91, p.height])
    return axes, counts, norm


# =================================================================== panel e
def hpd(v, mass=0.95):
    v = np.sort(np.asarray(v, float))
    v = v[np.isfinite(v)]
    k = int(np.floor(mass * len(v)))
    i = int(np.argmin(v[k:] - v[:len(v) - k]))
    return float(v[i]), float(v[i + k])


def wave_colours(norm, cmap=None, labels=None, periods=None):
    """Wave colour = the date scale of panel d, sampled at the wave midpoint,
    so colour means the same thing in both panels. `labels` supplies the keys
    (the dispersal tables and the map periods spell the waves differently);
    they are matched to the periods in order."""
    from matplotlib import cm
    periods = periods or v7.PHYLO_PERIODS
    cmap = cmap or cm.plasma
    mids = [0.5 * (a + b) for a, b in periods.values()]
    keys = list(labels) if labels is not None else [l.replace('\n', ' ') for l in periods]
    if len(keys) != len(mids):
        raise ValueError(f'{len(keys)} wave labels but {len(mids)} periods')
    return {k: cmap(norm(m)) for k, m in zip(keys, mids)}


def _violin(ax, x, vals, colour, width=0.34, lw=0.4):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    lo, hi = hpd(v)
    grid = np.linspace(v.min(), v.max(), 200)
    d = gaussian_kde(v).evaluate(grid)
    d = d / d.max() * width
    ax.fill_betweenx(grid, x - d, x + d, facecolor=colour, edgecolor='black',
                     linewidth=lw, alpha=0.9, zorder=3)
    med = float(np.median(v))
    ax.plot([x - width * 0.92, x + width * 0.92], [med, med], color='black',
            lw=0.8, solid_capstyle='butt', zorder=5)
    ax.plot([x, x], [lo, hi], color='black', lw=0.5, zorder=4)
    return med, lo, hi


def _stat_axis(ax, waves, colours, data, equal_data, *, ylabel, fmt, title,
               pp=None, log=False):
    meds = {}
    for i, w in enumerate(waves):
        med, lo, hi = _violin(ax, i, data[w], colours[w])
        meds[w] = (med, lo, hi)
        if equal_data is not None and w in equal_data:
            elo, ehi = hpd(equal_data[w])
            ax.plot([i + 0.46, i + 0.46], [elo, ehi], color='0.45', lw=0.5, zorder=4)
            ax.plot([i + 0.46], [np.median(equal_data[w])], 'o', ms=1.8,
                    mfc='white', mec='0.35', mew=0.45, zorder=5)
    # value labels go above each violin, then the brackets are lifted clear of
    # them, so neither can land on the densities or on the y-axis ticks
    span0 = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, w in enumerate(waves):
        med, lo, hi = meds[w]
        ax.text(i, hi + span0 * 0.025, fmt(med), ha='center', va='bottom',
                fontsize=FS_TICK, color='0.15', zorder=6)
    ax.set_xticks(range(len(waves)))
    ax.set_xticklabels([f'Wave {i + 1}' for i in range(len(waves))], fontsize=FS_TICK)
    ax.set_xlim(-0.78, len(waves) - 0.28)
    ax.set_ylabel(ylabel, fontsize=FS_ANN, labelpad=2)
    ax.set_title(title, fontsize=FS_BASE, loc='left', pad=3)
    ax.tick_params(length=2.0, pad=1.4)
    set_frame(ax)
    if log:
        ax.set_yscale('log')
    if pp:
        y0, y1 = ax.get_ylim()
        span = y1 - y0
        for (i, j), p in pp.items():
            yy = max(meds[waves[i]][2], meds[waves[j]][2]) + span * 0.135
            ax.plot([i, i, j, j], [yy - span * 0.022, yy, yy, yy - span * 0.022],
                    color='0.45', lw=0.45, zorder=6)
            ax.text((i + j) / 2, yy + span * 0.012, f'P = {p:.2f}', ha='center',
                    va='bottom', fontsize=FS_TICK - 0.5, color='0.35', zorder=6)
        ax.set_ylim(y0, y1 + span * 0.26)
    return meds


def panel_e(fig, subspec, W, Eq, WF, colours, *, periods=None):
    """Dispersal statistics the discrete model cannot produce: weighted lineage
    dispersal velocity, weighted diffusion coefficient, and wavefront distance
    from the epidemic origin through time."""
    periods = periods or v7.PHYLO_PERIODS
    waves = list(W.wave.unique())
    inner = subspec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.40)
    ax1, ax2, ax3 = [fig.add_subplot(inner[0, i]) for i in range(3)]

    vel = {w: g.velocity.values for w, g in W.groupby('wave')}
    dif = {w: g.diffusion.values for w, g in W.groupby('wave')}
    evel = {w: g.velocity.values for w, g in Eq.groupby('wave')}
    edif = {w: g.diffusion.values for w, g in Eq.groupby('wave')}

    pv = W.pivot(index='tree', columns='wave', values='velocity')
    pd_ = W.pivot(index='tree', columns='wave', values='diffusion')
    ppv = {(0, 1): float((pv[waves[1]] > pv[waves[0]]).mean()),
           (1, 2): float((pv[waves[2]] > pv[waves[1]]).mean())}
    ppd = {(0, 1): float((pd_[waves[1]] > pd_[waves[0]]).mean()),
           (1, 2): float((pd_[waves[2]] > pd_[waves[1]]).mean())}

    mv = _stat_axis(ax1, waves, colours, vel, evel,
                    ylabel='Dispersal velocity (km per year)',
                    fmt=lambda v: f'{v:.0f}', title='Dispersal velocity', pp=ppv)
    md = _stat_axis(ax2, waves, colours, dif, edif,
                    ylabel='Diffusion coefficient (km\u00b2 per year)',
                    fmt=lambda v: f'{v / 1e3:.1f}\u00d710\u00b3',
                    title='Diffusion coefficient', pp=ppd)

    # wavefront distance through time, with the wave windows marked; the
    # window colours are keyed positionally to the same waves as the densities
    for (lab, (a, b)), wkey in zip(periods.items(), waves):
        ax3.axvspan(a, b, color=colours[wkey], alpha=0.10, lw=0, zorder=0)
        ax3.text(0.5 * (a + b), 0.985, lab.split('\n')[0].replace('Wave ', 'W'),
                 transform=ax3.get_xaxis_transform(), ha='center', va='top',
                 fontsize=FS_TICK - 0.5, color='0.4')
    ax3.fill_between(WF.time, WF.lower, WF.upper, facecolor='0.6', alpha=0.30,
                     lw=0, zorder=2)
    ax3.plot(WF.time, WF['median'], color='0.15', lw=0.9, zorder=3)
    ax3.set_xlim(WF.time.min(), WF.time.max())
    ax3.set_ylim(0, WF.upper.max() * 1.06)
    ax3.set_xlabel('Year', fontsize=FS_ANN, labelpad=1.5)
    ax3.set_ylabel('Distance from origin (km)', fontsize=FS_ANN, labelpad=2)
    ax3.set_title('Wavefront distance', fontsize=FS_BASE, loc='left', pad=3)
    ax3.set_xticks(range(2018, 2027, 2))
    ax3.tick_params(length=2.0, pad=1.4, labelsize=FS_TICK)
    set_frame(ax3)

    summary = []
    for w in waves:
        v_m, v_lo, v_hi = mv[w]
        d_m, d_lo, d_hi = md[w]
        evlo, evhi = hpd(evel[w])
        edlo, edhi = hpd(edif[w])
        summary.append(dict(wave=w, n_trees=int((W.wave == w).sum()),
                            branches_min=int(W.loc[W.wave == w, 'nBranches'].min()),
                            branches_max=int(W.loc[W.wave == w, 'nBranches'].max()),
                            velocity_median=v_m, velocity_hpd_lo=v_lo, velocity_hpd_hi=v_hi,
                            velocity_equal_n_lo=evlo, velocity_equal_n_hi=evhi,
                            diffusion_median=d_m, diffusion_hpd_lo=d_lo, diffusion_hpd_hi=d_hi,
                            diffusion_equal_n_lo=edlo, diffusion_equal_n_hi=edhi))
    pp_rows = [dict(statistic='velocity', comparison=f'{waves[i]} vs {waves[j]}',
                    posterior_prob_increase=p) for (j, i), p in
               [((0, 1), ppv[(0, 1)]), ((1, 2), ppv[(1, 2)])]]
    pp_rows += [dict(statistic='diffusion', comparison=f'{waves[i]} vs {waves[j]}',
                     posterior_prob_increase=p) for (j, i), p in
                [((0, 1), ppd[(0, 1)]), ((1, 2), ppd[(1, 2)])]]
    return [ax1, ax2, ax3], pd.DataFrame(summary), pd.DataFrame(pp_rows)


# ================================================================== composite
def build_figure6(data, figsize=(7.2, 9.6)):
    """Assemble panels a-e at Nature double-column width."""
    from matplotlib import cm
    apply_style()
    fig = plt.figure(figsize=figsize, facecolor='w')
    gs = fig.add_gridspec(4, 1, height_ratios=[1.24, 1.18, 1.14, 0.98],
                          hspace=0.34, left=0.085, right=0.982,
                          top=0.978, bottom=0.042)

    ax_a = panel_a(fig, gs[0], data['r_wide'])
    letter(ax_a[0][0], 'a', x=-0.070, y=1.13)

    mid = gs[1].subgridspec(1, 2, wspace=0.34)
    gb = mid[0, 0].subgridspec(1, 2, width_ratios=[0.30, 1.0], wspace=0.0)
    gc = mid[0, 1].subgridspec(1, 2, width_ratios=[0.22, 1.0], wspace=0.0)
    axB = fig.add_subplot(gb[0, 1])
    axC = fig.add_subplot(gc[0, 1])
    panel_routes(axB, *data['s_late'][:3], level='state',
                 period_label=data['period_label'], max_x=data['s_late'][3])
    panel_routes(axC, *data['r_late'][:3], level='region',
                 period_label=data['period_label'], max_x=data['r_late'][3],
                 show_key=False)
    letter(axB, 'b', x=-0.40, y=1.03)
    letter(axC, 'c', x=-0.30, y=1.03)

    ax_d, counts, norm = panel_d(fig, gs[2], data['edge'], data['bg'],
                                 data['xlim'], data['ylim'])
    letter(ax_d[0], 'd', x=-0.02, y=1.00)

    colours = wave_colours(norm, cm.plasma, labels=list(data['W'].wave.unique()))
    ax_e, summary, pp = panel_e(fig, gs[3], data['W'], data['Eq'], data['WF'], colours)
    letter(ax_e[0], 'e', x=-0.26, y=1.04)
    return fig, dict(branch_counts=counts, summary=summary, posterior_probs=pp,
                     wave_colours=colours, norm=norm)


# ============================================================= extended data
def build_extended(data, figsize=(7.2, 9.6)):
    """Extended Data: pre-2024 state introductions and routes, plus the state
    and zone reference map moved out of Figure 6."""
    apply_style()
    fig = plt.figure(figsize=figsize, facecolor='w')
    gs = fig.add_gridspec(3, 1, height_ratios=[2.00, 1.05, 1.15], hspace=0.20,
                          left=0.085, right=0.982, top=0.975, bottom=0.030)

    ax_a = panel_a(fig, gs[0], data['s_pre_wide'], level='state',
                   facet_order=None, n_facets=8, late=False, dest_level='region',
                   full_xlim=data['pre_xlim'], first_case=2017.71)
    letter(ax_a[0][0], 'a', x=-0.070, y=1.10)

    mid = gs[1].subgridspec(1, 2, wspace=0.34)
    gb = mid[0, 0].subgridspec(1, 2, width_ratios=[0.30, 1.0], wspace=0.0)
    gc = mid[0, 1].subgridspec(1, 2, width_ratios=[0.22, 1.0], wspace=0.0)
    axB = fig.add_subplot(gb[0, 1])
    axC = fig.add_subplot(gc[0, 1])
    panel_routes(axB, *data['s_pre'][:3], level='state',
                 period_label=data['pre_label'], max_x=data['s_pre'][3])
    panel_routes(axC, *data['r_pre'][:3], level='region',
                 period_label=data['pre_label'], max_x=data['r_pre'][3],
                 show_key=False)
    letter(axB, 'b', x=-0.40, y=1.03)
    letter(axC, 'c', x=-0.30, y=1.03)

    ref = gs[2].subgridspec(2, 1, height_ratios=[1.0, 0.13], hspace=0.02)
    axM = fig.add_subplot(ref[0, 0])
    axL = fig.add_subplot(ref[1, 0])
    rx, ry = v7.reference_map_limits(data['bg'])
    v7.draw_region_reference_map(axM, data['bg'], title='', xlim=rx, ylim=ry)
    axM.set_title('States and geopolitical zones', fontsize=FS_BASE, loc='left',
                  pad=2, x=0.02)
    v7.draw_region_reference_legend(axL)
    letter(axM, 'd', x=-0.055, y=1.00)
    return fig


# ============================================================== script entry
PRE_PERIOD = (2015.00, 2024.00)      # complement of v7.LATE_PERIOD
PRE_ROUTE_THRESHOLD = 0.5            # min posterior mean per tree, pre-2024 routes
WAVES = {'Wave 1': (2015.90, 2021.00), 'Wave 2': (2021.00, 2024.00),
         'Wave 3': (2024.00, 2025.80)}


def _find(root, name):
    """Locate a data file anywhere under `root` (the folder keeps the discrete
    and continuous analyses in subdirectories)."""
    direct = os.path.join(root, name)
    if os.path.exists(direct):
        return direct
    for dirpath, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f'{name} not found under {root}')


def load_all(data_dir):
    """Read every input both figures need. Returns one dict."""
    d = os.path.abspath(data_dir).rstrip('/') + '/'
    v7.BASE = d
    v7.STATE_CSV = _find(d, 'State+Mpox+2026+results.tsv')
    v7.REGION_CSV = _find(d, '2026+Region+NCDC+MPOX.tsv')
    v7.GEOJSON = _find(d, 'gadm41_NGA_1.json')
    v7.PHYLO_BRANCH_CSV = _find(d, 'NCDC_MPOX_phylogeography_branches.csv')

    s_ev, s_ids = v7.load_markov_jumps(v7.STATE_CSV, 'state')
    r_ev, r_ids = v7.load_markov_jumps(v7.REGION_CSV, 'region')
    s_pre_ev = s_ev[(s_ev.time >= PRE_PERIOD[0]) & (s_ev.time < PRE_PERIOD[1])]

    bg = v7.load_nigeria_map(v7.GEOJSON)
    edge = v7.load_branch_csv(v7.PHYLO_BRANCH_CSV)
    xlim, ylim = v7.map_limits(bg)

    W = pd.read_csv(_find(d, 'dispersal_statistics_by_wave.csv'))
    Eq = pd.read_csv(_find(d, 'dispersal_statistics_by_wave_equal_n.csv'))
    WF = pd.read_csv(_find(d, 'wavefront_distance_through_time.csv'))

    jumps = []
    for lvl, ev, ids in (('state', s_ev, s_ids), ('region', r_ev, r_ids)):
        for w, (a, b) in WAVES.items():
            per = (ev[(ev.time >= a) & (ev.time < b)].groupby('treeId').size()
                   .reindex(ids, fill_value=0))
            lo, hi = hpd(per.values)
            jumps.append(dict(level=lvl, wave=w, median=per.median(), mean=per.mean(),
                              hpd_lo=lo, hpd_hi=hi, per_year=per.median() / (b - a)))

    return dict(
        r_wide=v7.build_timing_wide(r_ev, mean_threshold=v7.MEAN_THRESHOLD),
        s_late=v7.prep_late(s_ev, s_ids, 'state', v7.LATE_PERIOD),
        r_late=v7.prep_late(r_ev, r_ids, 'region', v7.LATE_PERIOD),
        s_pre_wide=v7.build_timing_wide(s_pre_ev, mean_threshold=PRE_ROUTE_THRESHOLD),
        s_pre=v7.prep_late(s_ev, s_ids, 'state', PRE_PERIOD),
        r_pre=v7.prep_late(r_ev, r_ids, 'region', PRE_PERIOD),
        bg=bg, edge=edge, xlim=xlim, ylim=ylim, W=W, Eq=Eq, WF=WF,
        jumps_by_wave=pd.DataFrame(jumps),
        period_label=f'{v7.LATE_PERIOD[0]:.0f}\u20132025', pre_label='pre-2024',
        pre_xlim=(2015.8, PRE_PERIOD[1] + 0.05))


def main(data_dir=None, out_stem='Figure6_MPXV_spatial'):
    d = data_dir or _HERE
    data = load_all(d)
    fig, meta = build_figure6(data)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{out_stem}.{ext}', **({'dpi': 400} if ext == 'png' else {}))
    figx = build_extended(data)
    for ext in ('png', 'pdf', 'svg'):
        figx.savefig(f'ExtendedData_MPXV_prewave3.{ext}',
                     **({'dpi': 400} if ext == 'png' else {}))
    meta['summary'].to_csv('dispersal_statistics_summary.csv', index=False)
    meta['posterior_probs'].to_csv('dispersal_posterior_probabilities.csv', index=False)
    data['jumps_by_wave'].to_csv('jumps_by_wave.csv', index=False)
    rep = layout_report(fig)
    print(f'wrote {out_stem}.png/.pdf/.svg and ExtendedData_MPXV_prewave3.* from {d}')
    print(f"branches per wave: {meta['branch_counts']}")
    print(f"layout: {rep['n_text']} text objects, "
          f"{len(rep['out_of_canvas'])} off-canvas")
    return fig, figx


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main(*sys.argv[1:3])
