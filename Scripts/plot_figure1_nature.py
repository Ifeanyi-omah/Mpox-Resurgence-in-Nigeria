#!/usr/bin/env python3
"""
plot_figure1.py - rebuild Figure 1 from the exported CSV tables.

Usage
    python plot_figure1.py                          # reads ./*.csv, writes Figure_1.png/.pdf
    python plot_figure1.py --data-dir path/to/csvs --out Figure_1_final
    python plot_figure1.py --geojson gadm41_NGA_1.json
    python plot_figure1.py --no-map                 # skip panel b if you have no geometry
    python plot_figure1.py --formats png pdf svg --dpi 400

Requires: pandas, numpy, matplotlib.  geopandas is needed only for the maps in panel b;
without it (or with --no-map) every other panel still renders.

Panel b needs Nigeria state geometry (GADM 4.1 level 1). Put gadm41_NGA_1.json in the data
directory, or pass --geojson, or let the script fetch it once with --download-geojson.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

GADM_URL = 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_NGA_1.json.zip'

ZONE_ORDER = ['South South', 'South East', 'South West',
              'North Central', 'North West', 'North East']
ZONE_COLORS = {'South West': '#56B4E9', 'South South': '#E69F00', 'South East': '#009E73',
               'North Central': '#7E6AA2', 'North West': '#D55E00', 'North East': '#D7B600'}
NORTH = ['North Central', 'North West', 'North East']
AGE_BANDS_PUB = ['0-10', '11-20', '21-30', '31-40', '41-50', '>50']
PYRAMID_AGES = ['0-4', '5-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-59', '60+']
POS_BANDS = ['0-4', '5-9', '10-17', '18-29', '30-44', '45+']

FILES = {
    'summary': '00_wave_summary.csv',
    'monthly': '01_panelA_monthly_cases_by_zone.csv',
    'genomes': '02_panelA_genomes_per_month_by_zone.csv',
    'states': '03_panelB_state_counts_by_wave.csv',
    'zones': '04_panelC_zone_composition_by_wave.csv',
    'ages': '05_panelD_age_composition_by_wave.csv',
    'pyramid': '06_panelE_wave3_age_sex_pyramid.csv',
    'positivity': '07_panelF_wave3_positivity_by_age_sex.csv',
}


# --------------------------------------------------------------------------- setup
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='.', help='directory holding the CSV tables (default: .)')
    p.add_argument('--out', default='Figure_1', help='output filename stem (default: Figure_1)')
    p.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                   choices=['png', 'pdf', 'svg', 'tif', 'eps'], help='output formats')
    p.add_argument('--dpi', type=int, default=300, help='raster resolution (default: 300)')
    p.add_argument('--geojson', default=None, help='path to gadm41_NGA_1.json')
    p.add_argument('--download-geojson', action='store_true',
                   help='fetch the GADM file into --data-dir if it is missing')
    p.add_argument('--no-map', action='store_true', help='skip panel b entirely')
    return p.parse_args()


def load_tables(data_dir):
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        sys.exit('Missing CSV table(s) in %s:\n  %s' % (data_dir, '\n  '.join(missing)))
    d = {k: pd.read_csv(os.path.join(data_dir, f)) for k, f in FILES.items()}
    d['monthly']['month'] = pd.to_datetime(d['monthly']['month'])
    d['genomes']['month'] = pd.to_datetime(d['genomes']['month'])
    return d


def find_geometry(args):
    """Return a GeoDataFrame of Nigerian states, or None."""
    if args.no_map:
        return None
    try:
        import geopandas as gpd
    except ImportError:
        print('! geopandas not installed - panel b will be blank. '
              'pip install geopandas, or pass --no-map', file=sys.stderr)
        return None

    candidates = [args.geojson] if args.geojson else []
    candidates += [os.path.join(args.data_dir, n) for n in
                   ('gadm41_NGA_1.json', 'gadm41_NGA_1.json.zip', 'nigeria_states.geojson')]
    path = next((c for c in candidates if c and os.path.exists(c)), None)

    if path is None and args.download_geojson:
        import urllib.request
        path = os.path.join(args.data_dir, 'gadm41_NGA_1.json.zip')
        print('Downloading state geometry from %s' % GADM_URL)
        try:
            urllib.request.urlretrieve(GADM_URL, path)
        except Exception as exc:                                   # noqa: BLE001
            print('! download failed (%s) - panel b will be blank' % exc, file=sys.stderr)
            return None

    if path is None:
        print('! No state geometry found - panel b will be blank. Put gadm41_NGA_1.json in %s, '
              'pass --geojson, or run with --download-geojson' % args.data_dir, file=sys.stderr)
        return None

    gdf = gpd.read_file(path)
    key = (gdf['NAME_1'].str.upper().str.replace('-', '', regex=False)
           .str.replace(' ', '', regex=False))
    fixes = {'AKWAIBOM': 'AKWA IBOM', 'CROSSRIVER': 'CROSS RIVER', 'CROSSRIVERS': 'CROSS RIVER',
             'FEDERALCAPITALTERRITORY': 'FCT', 'ABUJA': 'FCT', 'NASSARAWA': 'NASARAWA'}
    gdf['state'] = key.map(lambda k: fixes.get(k, k))
    gdf = gdf.dissolve(by='state', as_index=False)[['state', 'geometry']]
    cent = gdf.to_crs(3857).representative_point().to_crs(4326)
    return gdf.assign(cx=cent.x.values, cy=cent.y.values)


def style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 5, 'ytick.labelsize': 5, 'legend.fontsize': 6,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


# --------------------------------------------------------------------------- panels
def panel_a(axA, axG, tab, waves, spans):
    months = pd.date_range('2017-01-01', '2025-10-01', freq='MS')
    x = mpl.dates.date2num(months + pd.offsets.Day(14))
    width, YMAX = 24, 470

    for w in waves:                       # thin wave separators instead of grey blocks
        a, b = spans[w]
        for edge in (a, b + pd.Timedelta(days=1)):
            for ax_ in (axA, axG):
                ax_.axvline(mpl.dates.date2num(edge), color='#CCCCCC', lw=0.42,
                            linestyle=(0, (4, 4)), zorder=1)
    for w in waves:
        a, b = spans[w]
        axA.text(mpl.dates.date2num(a + (b - a) / 2), 1.02, w, transform=axA.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=6, color='#555555')

    m = tab['monthly']
    zoned = m[m['zone'].isin(ZONE_ORDER)]
    bottom = np.zeros(len(months))
    for z in ZONE_ORDER:
        sub = zoned[zoned['zone'] == z].groupby('month')[['confirmed', 'suspected_not_confirmed']].sum()
        conf = sub['confirmed'].reindex(months).fillna(0).values
        non = sub['suspected_not_confirmed'].reindex(months).fillna(0).values
        axA.bar(x, conf, width=width, bottom=bottom, color=ZONE_COLORS[z],
                edgecolor='white', linewidth=0.3, zorder=3)
        bottom += conf
        axA.bar(x, non, width=width, bottom=bottom, color=ZONE_COLORS[z], alpha=0.25,
                edgecolor=ZONE_COLORS[z], hatch='////', linewidth=0.3, zorder=2)
        bottom += non

    nat = m[~m['zone'].isin(ZONE_ORDER)]           # 2023: national only, no state breakdown
    if len(nat):
        axA.bar(mpl.dates.date2num(nat['month'] + pd.offsets.Day(14)), nat['confirmed'],
                width=width, color='#8C8C8C', edgecolor='white', linewidth=0.3, zorder=3)
        axA.annotate('2023 confirmed cases\n(no state breakdown\nin the sitrep)',
                     xy=(mpl.dates.date2num(nat['month'].min() + pd.offsets.Day(14)),
                         float(nat['confirmed'].max())),
                     xytext=(mpl.dates.date2num(pd.Timestamp('2023-05-01')), 165),
                     fontsize=5, color='#666666', linespacing=1.2, ha='left', va='center',
                     arrowprops=dict(arrowstyle='-', color='#999999', lw=0.38), zorder=6)

    axA.set_ylabel('Monthly cases')
    axA.set_ylim(0, YMAX)
    axA.yaxis.set_major_locator(MultipleLocator(100))
    axA.grid(axis='y', color='#D9D9D9', linewidth=0.34, alpha=0.8, zorder=0)
    axA.spines[['top', 'right']].set_visible(False)
    axA.set_xlim(mpl.dates.date2num(pd.Timestamp('2016-12-01')),
                 mpl.dates.date2num(pd.Timestamp('2025-11-01')))
    years = pd.date_range('2017-01-01', '2025-01-01', freq='YS')
    axA.set_xticks(mpl.dates.date2num(years))
    axA.set_xticklabels([str(y.year) for y in years])
    axA.tick_params(axis='x', labelbottom=False, length=0)

    leg1 = axA.legend(handles=[
        Patch(facecolor='#4D4D4D', edgecolor='none', label='Confirmed (PCR)'),
        Patch(facecolor='#BEBEBE', edgecolor='#7A7A7A', hatch='////', alpha=0.5,
              label='Suspected, not confirmed')],
        loc='upper left', bbox_to_anchor=(0.005, 1.0), frameon=False, handlelength=1.4,
        handleheight=1.2, handletextpad=0.5, borderpad=0.1, labelspacing=0.3, fontsize=6)
    axA.add_artist(leg1)
    axA.legend(handles=[Line2D([0], [0], marker='s', linestyle='None', markersize=5.0,
                               markerfacecolor=ZONE_COLORS[z], markeredgecolor=ZONE_COLORS[z],
                               label=z) for z in ZONE_ORDER],
               ncol=3, loc='upper left', bbox_to_anchor=(0.175, 1.0), frameon=False,
               columnspacing=1.2, handletextpad=0.35, borderpad=0.1, labelspacing=0.3, fontsize=5)

    g = tab['genomes']
    gb = np.zeros(len(months))
    for z in ZONE_ORDER:
        vals = (g[g['zone'] == z].groupby('month')['genomes'].sum()
                .reindex(months).fillna(0).values)
        axG.bar(x, vals, width=width, bottom=gb, color=ZONE_COLORS[z],
                edgecolor='white', linewidth=0.3, zorder=3)
        gb += vals
    mx = max(20, int(math.ceil(gb.max() / 5) * 5))
    axG.set_ylim(mx, 0)
    axG.set_ylabel('Genomes', fontsize=6)
    axG.set_yticks(np.arange(0, mx + 1, 10))
    axG.grid(axis='y', color='#E5E5E5', linewidth=0.3, alpha=0.8, zorder=0)
    axG.spines[['top', 'right']].set_visible(False)
    axG.tick_params(axis='x', which='both', bottom=True, labelbottom=True, pad=3.2)
    axG.xaxis.set_ticks_position('bottom')


def panel_b(map_axes, cax, fig, tab, waves, geo):
    st = tab['states']
    if geo is None:
        for ax, w in zip(map_axes, waves):
            ax.set_axis_off()
            ax.text(0.5, 0.5, '%s\n(no geometry available)' % w, ha='center', va='center',
                    fontsize=6, color='#999999', transform=ax.transAxes)
        cax.set_axis_off()
        return

    zone_geo = geo.assign(zone=geo['state'].map(
        {s: z for z, ss in _zone_states().items() for s in ss})).dropna(subset=['zone'])
    zone_geo = zone_geo.dissolve(by='zone', as_index=False)

    vmax = float(np.ceil(st['pct_of_wave_confirmed'].max() / 5) * 5)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = mpl.colormaps['YlOrBr']
    conf_max = float(st['confirmed'].max())

    for ax, w in zip(map_axes, waves):
        sub = st[st['wave'] == w][['state', 'confirmed', 'pct_of_wave_confirmed']]
        g = geo.merge(sub, on='state', how='left')
        g[['confirmed', 'pct_of_wave_confirmed']] = g[['confirmed', 'pct_of_wave_confirmed']].fillna(0)
        g.plot(ax=ax, column='pct_of_wave_confirmed', cmap=cmap, norm=norm,
               edgecolor='white', linewidth=0.34)
        zero = g[g['confirmed'] == 0]
        if len(zero):
            zero.plot(ax=ax, color='#F2F2F2', edgecolor='white', linewidth=0.34)
        geo.boundary.plot(ax=ax, color='#9A9A9A', linewidth=0.3)
        zone_geo.boundary.plot(ax=ax, color='#3A3A3A', linewidth=0.59, alpha=0.7)
        gc = g[g['confirmed'] > 0]
        ax.scatter(gc['cx'], gc['cy'], s=18 + 620 * gc['confirmed'] / conf_max,
                   facecolor='none', edgecolor='#2B2B2B', linewidth=0.59, zorder=6)
        ax.set_axis_off(); ax.set_aspect('equal')
        ax.set_xlim(2.4, 15.0); ax.set_ylim(3.9, 14.2)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_label("State's share of that wave's confirmed cases (%)", labelpad=1.6, fontsize=6)
    cb.ax.tick_params(labelsize=5, length=1.6, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.8)
    map_axes[2].legend(handles=[
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none',
               markeredgecolor='#2B2B2B', markeredgewidth=0.59,
               markersize=np.sqrt(18 + 620 * n / conf_max) * 0.95, label=str(n))
        for n in (10, 50, 150)],
        title='confirmed cases', loc='upper right', bbox_to_anchor=(1.10, 0.42), frameon=False,
        fontsize=5, title_fontsize=5, labelspacing=1.1, handletextpad=1.0, borderpad=0.2)


def zone_bar(ax, tab, wave, first=False, title=True):
    """100% stacked zone-composition bar sitting on top of one wave map."""
    z = tab['zones']
    z = z[z['wave'] == wave].set_index('zone')['pct_of_wave_confirmed'].reindex(ZONE_ORDER).fillna(0)
    left = 0.0
    for zone in ZONE_ORDER:
        v = float(z[zone])
        ax.barh([0], [v], left=left, height=1.0, color=ZONE_COLORS[zone],
                edgecolor='white', linewidth=0.42)
        if v >= 8:
            ax.text(left + v / 2, 0, f'{v:.0f}', ha='center', va='center',
                    fontsize=5, color='white', fontweight='bold')
        left += v
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 0.5)
    ax.set_axis_off()
    if title:
        ax.set_title(wave, fontsize=7, pad=13.6)
    if first:
        ax.text(0, 1.30, 'Confirmed cases by zone (%)', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=5, color='#555555')


def panel_d(axE, tab, waves):
    a = tab['ages'].pivot(index='age_band', columns='wave',
                          values='pct_of_wave_confirmed').reindex(AGE_BANDS_PUB)
    n = tab['ages'].groupby('wave')['confirmed'].sum()
    fill = dict(zip(waves, ['#C9C9C9', '#7FA9C0', '#2E6B8C']))
    edge = dict(zip(waves, ['#6E6E6E', '#3F7FA6', '#17455C']))
    bw, xs = 0.27, np.arange(len(AGE_BANDS_PUB))
    for i, w in enumerate(waves):
        vals = a[w].values
        axE.bar(xs + (i - 1) * bw, vals, bw, color=fill[w], edgecolor=edge[w], linewidth=0.42,
                label='%s (n=%d)' % (w, n[w]))
    axE.set_xticks(xs); axE.set_xticklabels(AGE_BANDS_PUB, fontsize=5)
    axE.set_xlabel('Age group (years)'); axE.set_ylabel('Confirmed cases (%)')
    axE.set_ylim(0, 50)
    axE.yaxis.set_major_locator(MultipleLocator(10))
    axE.grid(axis='y', color='#E5E5E5', linewidth=0.34)
    axE.spines[['top', 'right']].set_visible(False)
    axE.legend(frameon=False, fontsize=5, loc='upper right', bbox_to_anchor=(1.03, 1.04))


def panel_e(axD, tab):
    p = tab['pyramid']

    def get(age, sex, status):
        m = p[(p['age_group'] == age) & (p['sex'] == sex) & (p['status'] == status)]
        return int(m['n'].sum())

    y = np.arange(len(PYRAMID_AGES))
    fn = np.array([get(a, 'FEMALE', 'suspected_not_confirmed') for a in PYRAMID_AGES])
    fc = np.array([get(a, 'FEMALE', 'confirmed') for a in PYRAMID_AGES])
    mn = np.array([get(a, 'MALE', 'suspected_not_confirmed') for a in PYRAMID_AGES])
    mc = np.array([get(a, 'MALE', 'confirmed') for a in PYRAMID_AGES])
    axD.barh(y, -fn, color='#E4C7E8', edgecolor='white', linewidth=0.3, height=0.78,
             label='Female (suspected)')
    axD.barh(y, -fc, left=-fn, color='#8E4B9D', edgecolor='white', linewidth=0.3, height=0.78,
             label='Female (confirmed)')
    axD.barh(y, mn, color='#B9D6E5', edgecolor='white', linewidth=0.3, height=0.78,
             label='Male (suspected)')
    axD.barh(y, mc, left=mn, color='#3F7FA6', edgecolor='white', linewidth=0.3, height=0.78,
             label='Male (confirmed)')
    axD.axvline(0, color='#666666', linewidth=0.42)
    axD.set_yticks(y); axD.set_yticklabels(PYRAMID_AGES)
    axD.set_ylabel('Age group (years)'); axD.set_xlabel('Cases')
    axD.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{abs(int(v))}'))
    lim = int(math.ceil(max((fn + fc).max(), (mn + mc).max()) / 50.0) * 50 + 25)
    axD.set_xlim(-lim, lim); axD.set_ylim(-1.15, len(PYRAMID_AGES) - 0.25)
    axD.xaxis.set_major_locator(MultipleLocator(100))
    axD.grid(axis='x', color='#E5E5E5', linewidth=0.34)
    axD.spines[['top', 'right']].set_visible(False)
    axD.text(-lim * 0.60, -0.85, 'Female', ha='center', va='center', fontsize=7, color='#8E4B9D')
    axD.text(lim * 0.60, -0.85, 'Male', ha='center', va='center', fontsize=7, color='#3F7FA6')
    axD.legend(loc='upper right', bbox_to_anchor=(1.02, 1.02), frameon=False, fontsize=5,
               handlelength=1.1, labelspacing=0.2)
    axD.set_title('Wave 3', fontsize=7, pad=3.2, loc='left')


def panel_ref(axR, geo):
    """Reference map: Nigeria by geopolitical zone, with every state labelled."""
    if geo is None:
        axR.set_axis_off()
        axR.text(0.5, 0.5, 'Reference map\n(no geometry available)', ha='center', va='center',
                 fontsize=6, color='#999999', transform=axR.transAxes)
        return

    zone_of = {st: z for z, ss in _zone_states().items() for st in ss}
    g = geo.copy()
    g['zone'] = g['state'].map(zone_of)
    g['color'] = g['zone'].map(ZONE_COLORS).fillna('#EEEEEE')
    g.plot(ax=axR, color=g['color'], edgecolor='white', linewidth=0.46, alpha=0.92)
    zone_geo = g.dropna(subset=['zone']).dissolve(by='zone', as_index=False)
    zone_geo.boundary.plot(ax=axR, color='#2B2B2B', linewidth=0.92)

    pretty = {'AKWA IBOM': 'Akwa Ibom', 'CROSS RIVER': 'Cross River', 'FCT': 'FCT'}

    # crowded southern states get leader lines into the left/right margins
    left = ['EKITI', 'OSUN', 'OGUN', 'LAGOS', 'ONDO', 'EDO', 'DELTA', 'BAYELSA', 'RIVERS']
    right = ['ENUGU', 'EBONYI', 'ANAMBRA', 'IMO', 'ABIA', 'CROSS RIVER', 'AKWA IBOM']
    callout = {}
    for i, st in enumerate(left):
        callout[st] = (1.15, 10.3 - i * 0.78)
    for i, st in enumerate(right):
        callout[st] = (16.2, 10.6 - i * 0.85)

    for _, r in g.iterrows():
        st = r['state']
        name = pretty.get(st, st.title())
        if st in callout:
            tx, ty = callout[st]
            axR.annotate(name, xy=(r['cx'], r['cy']), xytext=(tx, ty),
                         ha='right' if tx < 8 else 'left', va='center',
                         fontsize=5, color='#111111',
                         arrowprops=dict(arrowstyle='-', color='#888888', lw=0.42,
                                         shrinkA=2, shrinkB=2),
                         bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                                   edgecolor='#C8C8C8', lw=0.3, alpha=0.95),
                         zorder=8)
        else:
            axR.text(r['cx'], r['cy'], name, ha='center', va='center', fontsize=5,
                     color='#111111', zorder=5,
                     path_effects=[pe.withStroke(linewidth=1.09, foreground='white')])

    axR.set_axis_off(); axR.set_aspect('equal')
    axR.set_xlim(0.4, 17.6); axR.set_ylim(3.6, 14.4)
    axR.legend(handles=[Patch(facecolor=ZONE_COLORS[z], edgecolor='white', label=z)
                        for z in ZONE_ORDER],
               loc='lower center', bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=6,
               ncol=3, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)


def panel_f(axF, tab):
    p = tab['positivity']
    xs = np.arange(len(POS_BANDS))
    for sex, colr, off, lab in [('MALE', '#3F7FA6', -0.19, 'Male'),
                                ('FEMALE', '#8E4B9D', 0.19, 'Female')]:
        vals = [float(p[(p['band'].astype(str) == b) & (p['sex'] == sex)]['pct_positive'].sum())
                for b in POS_BANDS]
        axF.bar(xs + off, vals, 0.36, color=colr, edgecolor='white', linewidth=0.3, label=lab)
    axF.set_ylim(0, 44)
    axF.axvline(2.5, color='#999999', linewidth=0.42, linestyle='--')
    for x0, x1, lab in [(-0.35, 2.35, 'children (<18 y)'), (2.65, 5.35, 'adults')]:
        axF.plot([x0, x1], [-0.13, -0.13], transform=axF.get_xaxis_transform(),
                 color='#888888', lw=0.5, clip_on=False)
        axF.text((x0 + x1) / 2, -0.155, lab, transform=axF.get_xaxis_transform(),
                 ha='center', va='top', fontsize=5, color='#555555')
    axF.set_xticks(xs); axF.set_xticklabels(POS_BANDS)
    axF.set_xlabel('Age group (years)', labelpad=16.8)
    axF.set_ylabel('PCR-positive among\nsuspected cases (%)')
    axF.grid(axis='y', color='#E5E5E5', linewidth=0.34)
    axF.spines[['top', 'right']].set_visible(False)
    axF.legend(frameon=False, fontsize=6, loc='upper left')
    axF.set_title('Wave 3', fontsize=7, pad=3.2, loc='left')


def _zone_states():
    return {
        'South West': ['LAGOS', 'OGUN', 'ONDO', 'OSUN', 'OYO', 'EKITI'],
        'South South': ['AKWA IBOM', 'BAYELSA', 'CROSS RIVER', 'DELTA', 'EDO', 'RIVERS'],
        'South East': ['ABIA', 'ANAMBRA', 'EBONYI', 'ENUGU', 'IMO'],
        'North Central': ['BENUE', 'FCT', 'KOGI', 'KWARA', 'NASARAWA', 'NIGER', 'PLATEAU'],
        'North West': ['JIGAWA', 'KADUNA', 'KANO', 'KATSINA', 'KEBBI', 'SOKOTO', 'ZAMFARA'],
        'North East': ['ADAMAWA', 'BAUCHI', 'BORNO', 'GOMBE', 'TARABA', 'YOBE'],
    }


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    style()
    tab = load_tables(args.data_dir)
    geo = find_geometry(args)

    waves = list(tab['summary']['wave'])
    spans = {r['wave']: (pd.Timestamp(r['period'].split(' to ')[0]),
                         pd.Timestamp(r['period'].split(' to ')[1]))
             for _, r in tab['summary'].iterrows()}

    fig = plt.figure(figsize=(7.2, 9.0), facecolor='white')
    axA = fig.add_axes([0.065, 0.912, 0.905, 0.066])
    axG = fig.add_axes([0.065, 0.858, 0.905, 0.034], sharex=axA)
    axR = fig.add_axes([0.520, 0.035, 0.460, 0.230])
    bar_axes = [fig.add_axes([x + 0.015, 0.775, 0.27, 0.017]) for x in (0.035, 0.345, 0.655)]
    map_axes = [fig.add_axes([x, 0.575, 0.30, 0.190]) for x in (0.035, 0.345, 0.655)]
    cax = fig.add_axes([0.240, 0.552, 0.17, 0.009])
    
    axE = fig.add_axes([0.080, 0.335, 0.370, 0.165])
    axD = fig.add_axes([0.590, 0.335, 0.370, 0.165])
    axF = fig.add_axes([0.080, 0.070, 0.330, 0.170])

    panel_a(axA, axG, tab, waves, spans)
    panel_b(map_axes, cax, fig, tab, waves, geo)
    panel_d(axE, tab, waves)
    panel_e(axD, tab)
    for i, (bax, w) in enumerate(zip(bar_axes, waves)):
        zone_bar(bax, tab, w, first=(i == 0))
    panel_ref(axR, geo)
    panel_f(axF, tab)

    for xx, yy, lab in [(0.015, 0.993, 'a'), (0.015, 0.832, 'b'), (0.015, 0.522, 'c'),
                        (0.525, 0.522, 'd'), (0.015, 0.268, 'e'), (0.470, 0.268, 'f')]:
        fig.text(xx, yy, lab, fontsize=8, fontweight='bold', ha='left', va='top')
    if geo is not None:
        fig.text(0.470, 0.5485, 'grey = no confirmed case in that wave', fontsize=5,
                 color='#555555', ha='left', va='center')
    for ext in args.formats:
        path = '%s.%s' % (args.out, ext)
        fig.savefig(path, facecolor='white',
                    **({'dpi': args.dpi} if ext in ('png', 'tif') else {}))
        print('Wrote %s (%.1f MB)' % (path, os.path.getsize(path) / 1e6))
    plt.close(fig)


if __name__ == '__main__':
    main()
