"""Build the MPXV Nigeria-Cameroon Figure 1.

Usage:  python build_figure.py [data_dir] [out_stem]

data_dir defaults to the folder this script lives in and must contain the two
trees, their branch_snps reconstruction CSVs, the curated metadata CSV and the
two GADM level-1 GeoJSON files. Writes <out_stem>.png/.pdf/.svg.
"""
import os
import sys
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

_HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _HERE)
from mpxvfig import *          # palettes, tree parser, layout, map drawing, load_data

L_ALN = 182679
TREE_W_B = 0.60      # panel b: fraction of the x-range given to the tree
FS_BASE, FS_ANN, FS_TICK = 9, 8, 7

def snp_edge_dict(df):
    d = {}
    for (p, ch), g in df.groupby(['parent', 'child']):
        d[(p, ch)] = list(g.apo.astype(bool))
    return d

def key_block(ax, x, y, entries, title=None, dy=0.0195, fs=None, sw=0.024):
    """Draw a frameless key in axes coords. entries: (kind, colour, edge, label)."""
    fs = fs or FS_ANN
    if title:
        ax.text(x, y, title, transform=ax.transAxes, fontsize=fs, weight='bold',
                va='center', ha='left', color='0.15'); y -= dy*1.15
    for kind, fc, ec, lab in entries:
        if kind == 'sq':
            ax.add_patch(Rectangle((x, y-dy*0.3), sw, dy*0.62, transform=ax.transAxes,
                                   facecolor=fc, edgecolor=ec or 'none', lw=0.3,
                                   clip_on=False, zorder=5))
        else:
            ax.plot([x+sw/2], [y], 'o', ms=4.4, mfc=fc, mec=ec or 'none', mew=0.5,
                    transform=ax.transAxes, clip_on=False, zorder=5)
        ax.text(x+sw+0.012, y, lab, transform=ax.transAxes, fontsize=fs,
                va='center', ha='left', color='0.15')
        y -= dy
    return y

# ============================================================ panel a
def panel_a(axt, axb):
    X, Y, order = layout(root, rev=False)
    tips = [nd.name for nd in order]
    meta = TM.set_index('tip').loc[tips]
    grp, lin, camf = meta.group.to_dict(), meta.linbar.to_dict(), meta.cameroon.to_dict()

    segs, cols, lws = branch_segments(nodes, X, Y, lambda p, nd: (BRANCH_C, 0.4))
    segs = [[(x*L_ALN, y) for x, y in s] for s in segs]
    axt.add_collection(LineCollection(segs, colors=BRANCH_C, linewidths=0.4,
                                      capstyle='round', zorder=2))
    xs, ys, cs = snp_dots(nodes, X, Y, snp_edge_dict(sn), nid, xscale=L_ALN, yoff=-0.52)
    axt.scatter(xs, ys, s=2.3, c=cs, linewidths=0, zorder=4)
    for g in ['pub', 'new', 'cam']:
        i = [j for j, t in enumerate(tips) if grp[t] == g]
        axt.scatter([X[id(order[j])]*L_ALN for j in i], i,
                    s=(5.0 if g == 'pub' else 11), facecolors=TIP[g],
                    edgecolors=TIP_EDGE[g], linewidths=0.4, zorder=5)
    xmax = max(X.values())*L_ALN
    axt.set_xlim(-4, xmax*1.02); axt.set_ylim(len(tips)+6, -12)
    axt.set_xticks([]); axt.set_yticks([]); set_frame(axt, 'none')
    axt.set_title('',
                  fontsize=FS_BASE, loc='left', pad=3, x=0.075)

    # scale bar
    x0 = xmax*0.52
    axt.plot([x0, x0+20], [len(tips)+2.5]*2, color='0.15', lw=0.9, solid_capstyle='butt')
    axt.text(x0+10, len(tips)+5.5, '7e-5 SNPs', ha='center', va='top',
             fontsize=FS_TICK, color='0.15')

    # lineage strip
    blocks, start = [], 0
    for i in range(1, len(tips)+1):
        if i == len(tips) or lin[tips[i]] != lin[tips[start]]:
            blocks.append((lin[tips[start]], start, i)); start = i
    for name, s, e in blocks:
        axb.add_patch(Rectangle((0, s-0.5), 1, e-s, facecolor=LIN[name],
                                edgecolor='0.45', linewidth=0.15))
    axb.set_xlim(0, 1); axb.set_ylim(len(tips)+6, -8)
    axb.set_xticks([]); axb.set_yticks([]); set_frame(axb, 'none')
    axb.text(0.5, 0.5*(blocks[0][1]+blocks[0][2]), 'Lineage', ha='center', va='center',
             fontsize=FS_TICK, color='0.15', rotation=90, zorder=6)

    # clade annotations
    nc = mrca(nodes, [t for t in tips if camf[t]])
    ncl = {l.name for l in leaves_of(nc)}
    i_nc = [i for i, t in enumerate(tips) if t in ncl]
    i_se, i = [], max(i_nc)+1              # contiguous zoonotic grade below that clade
    while i < len(tips) and meta.linbar[tips[i]] == 'zoonotic':
        i_se.append(i); i += 1
    hu = mrca(nodes, TM.loc[~TM.zoonotic, 'tip'])
    def bracket(rows, label, xb, extra=None):
        axt.plot([xb]*2, [min(rows)-0.5, max(rows)+0.5], color='0.15', lw=1.1,
                 solid_capstyle='butt', zorder=6)
        txt = label if not extra else f'{label}\n{extra}'
        axt.text(xb+xmax*0.02, 0.5*(min(rows)+max(rows)), txt, va='center', ha='left',
                 fontsize=FS_ANN, linespacing=1.2, zorder=6)
    bracket(i_nc, 'Nigeria–Cameroon clade', xmax*0.40, extra='(expanded in b)')
    bracket(i_se, 'South-East zoonotic', xmax*0.40)
    xh, yh = X[id(hu.parent)]*L_ALN, Y[id(hu)]
    axt.annotate('Emergence into\nSH',
                 xy=(xh, yh), xytext=(0, yh+62),
                 fontsize=FS_ANN, ha='left', va='center', linespacing=1.25,
                 arrowprops=dict(arrowstyle='-|>', lw=0.8, color='0.15',
                                 shrinkA=2, shrinkB=2, connectionstyle='arc3,rad=-0.25'))
    # keys
    y = key_block(axt, 0.015, 0.285,
                  [('o', TIP[g], TIP_EDGE[g], TIP_LABEL[g]) for g in ('new', 'pub', 'cam')],
                  title='Genome')
    y = key_block(axt, 0.015, y-0.014, [('o', APO_C, None, 'APOBEC3-context'),
                                        ('o', NONAPO_C, None, 'other')],
                  title='Substitution on branch')
    y -= 0.014
    ks = [('sq', LIN[k], None, LIN_LABEL[k]) for k in LIN_ORDER]
    key_block(axt, 0.015, y, ks[:5], title='Lineage')
    key_block(axt, 0.175, y - 0.0185*1.15, ks[5:])
    return tips, meta

# ============================================================ panel b
def panel_b(ax):
    r2, n2 = parse_nexus(D+"unannotated.tree")
    X, Y, order = layout(r2, rev=False)
    tips = [nd.name for nd in order]
    segs, cols, lws = branch_segments(n2, X, Y, lambda p, nd: (BRANCH_C, 0.55))
    xmax = max(X.values())/TREE_W_B
    segs = [[(x/xmax, y) for x, y in s] for s in segs]
    ax.add_collection(LineCollection(segs, colors=BRANCH_C, linewidths=0.55,
                                     capstyle='round', zorder=2))
    sx, sy, sc = snp_dots(n2, X, Y, snp_edge_dict(sn2), nid, xscale=1/xmax, yoff=-0.135)
    ax.scatter(sx, sy, s=7.0, c=sc, linewidths=0, zorder=4)
    fc = []
    for t in tips:
        reg, ctry, _ = pretty_tip(t)
        fc.append(TIP['cam'] if ctry == 'Cameroon' else STATE_C_PRETTY.get(reg, '#FFFFFF'))
    ax.scatter([X[id(order[j])]/xmax for j in range(len(tips))], range(len(tips)),
               s=18, facecolors=fc, edgecolors='0.15', linewidths=0.45, zorder=3)
    labs = []
    for j, t in enumerate(tips):
        reg, ctry, yr = pretty_tip(t)
        labs.append(ax.text(X[id(order[j])]/xmax + 0.016, j, f'{reg} | {ctry} | {yr}',
                            va='center', ha='left', fontsize=FS_TICK, color='0.1'))
    # solve the x-range in display units: label text width is independent of xlim,
    # so pick the right limit that lets the longest label clear the clade bracket
    rend = ax.figure.canvas.get_renderer()
    W = ax.get_window_extent(rend).width
    px = ax.figure.dpi/72.0
    pad = 26*px                                     # bracket line + rotated label
    x0 = -0.015
    need = max((X[id(order[j])]/xmax + 0.016 - x0) * W / max(W - pad - t.get_window_extent(rend).width, 1.0)
               for j, t in enumerate(labs))
    R = x0 + need
    ax.set_xlim(x0, R); ax.set_ylim(len(tips)-0.3, -1.2)
    d_per_px = (R - x0)/W
    xbr = x0 + (W - pad + 7*px)*d_per_px
    ax.set_xticks([]); ax.set_yticks([]); set_frame(ax, 'none')
    ax.set_title('',
                 fontsize=FS_BASE, loc='left', pad=3, x=0.03)
    # clade brackets
    camset = [t for t in tips if 'cameroon' in t.lower()]
    nc = mrca(n2, camset)
    ncl = {l.name for l in leaves_of(nc)}
    se = mrca(n2, [t for t in tips if t not in ncl])
    for nd, lab in [(se, 'South-East'), (nc, 'Nigeria–Cameroon')]:
        ys = [Y[id(l)] for l in leaves_of(nd)]
        ax.plot([xbr]*2, [min(ys), max(ys)], color='0.1', lw=2, solid_capstyle='butt')
        ax.text(xbr + 9*px*d_per_px, 0.5*(min(ys)+max(ys)), lab, rotation=90,
                va='center', ha='center', fontsize=FS_ANN)
    # scale bar in SNPs
    sb = 5/(xmax*L_ALN)
    ax.plot([0.015, 0.015+sb], [len(tips)-0.9]*2, color='0.15', lw=0.9,
            solid_capstyle='butt')
    ax.text(0.015+sb/2, len(tips)-1.6, '3e-5 SNPs', ha='center', va='bottom',
            fontsize=FS_TICK, color='0.15')

# ============================================================ panel c
def panel_c(ax):
    sub = TM[TM.group == 'new']
    order_c = [k for k in LIN_ORDER if (sub.linbar == k).any()]
    vals = [int((sub.linbar == k).sum()) for k in order_c]
    xs = np.arange(len(order_c))
    ax.bar(xs, vals, color=[LIN[k] for k in order_c], width=0.68,
           edgecolor='0.25', linewidth=0.4, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + max(vals)*0.03, str(v), ha='center', va='bottom',
                fontsize=FS_ANN, color='0.15')
    ax.set_xticks(xs)
    ax.set_xticklabels([LIN_LABEL[k] for k in order_c], fontsize=FS_TICK)
    ax.set_ylabel('Genomes (n)', fontsize=FS_BASE)
    ax.set_ylim(0, max(vals)*1.18)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    set_frame(ax, 'open')
    ax.set_title('',
                 fontsize=FS_BASE, loc='left', pad=3, x=0.075)

# ============================================================ panel d
SITES = [('Bamenda', 'NW', 5.9631, 10.1591), ('Njikwa', 'NW', 6.0854, 9.8934),
         ('Tombel', 'SW', 4.7451, 9.6703), ('Bangem', 'SW', 5.0862, 9.7665),
         ('Ndikiniméki', 'C', 4.7712, 10.8282), ('Benakuma', 'NW', 6.4139, 9.9144)]

PANEL_B_STATES = ['Abia', 'AkwaIbom', 'CrossRiver', 'Edo', 'Imo', 'Ondo']
STATE_PRETTY = {'AkwaIbom': 'Akwa Ibom', 'CrossRiver': 'Cross River'}
STATE_LABEL_XY = {'Ondo': (3.25, 3.60), 'Edo': (4.35, 2.60), 'Imo': (5.60, 3.60),
                  'Abia': (6.75, 2.60), 'AkwaIbom': (8.15, 3.60),
                  'CrossRiver': (9.25, 2.60)}
CMR_LABEL_XY = {'Nord-Ouest': (13.05, 7.65), 'Sud-Ouest': (12.65, 6.25),
                'Centre': (12.25, 4.25)}

def _halo(t, lw=1.6):
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground='white')])

def _ring_centroid(f):
    best, area = None, -1
    for poly in f['geometry']['coordinates']:
        r = np.asarray(poly[0])
        a = 0.5*abs(np.dot(r[:-1, 0], r[1:, 1]) - np.dot(r[1:, 0], r[:-1, 1]))
        if a > area: area, best = a, r
    return best.mean(axis=0)

def panel_d(ax):
    z_of = {s: z for z, ss in ZONES.items() for s in ss}
    ax.add_collection(geo_patches(ng['features'],
                                  lambda n: ZONE_C.get(z_of.get(n, ''), '#DDDDDD')))
    ax.add_collection(geo_patches(cm['features'],
                                  lambda n: CMR_SAMPLED if n in CMR_SAMPLED_REGIONS
                                  else CMR_UNSAMPLED))
    # states contributing genomes to b: own fill colour + heavier outline
    ax.add_collection(geo_patches([f for f in ng['features']
                                   if f['properties']['NAME_1'] in PANEL_B_STATES],
                                  lambda n: STATE_C[n], lw=0.7, ec='0.1'))
    for f in ng['features']:
        if f['properties']['NAME_1'] in PANEL_B_STATES:
            for poly in f['geometry']['coordinates']:
                r = np.asarray(poly[0])
                ax.plot(r[:, 0], r[:, 1], color='0.1', lw=0.7, zorder=4)
    for feats in (ng['features'], cm['features']):     # country outline
        for f in feats:
            for poly in f['geometry']['coordinates']:
                r = np.asarray(poly[0])
                ax.plot(r[:, 0], r[:, 1], color='0.35', lw=0.2, zorder=3)
    # labels: panel-b states and sampled Cameroon regions
    for f in ng['features']:
        nm = f['properties']['NAME_1']
        if nm in PANEL_B_STATES:
            cx, cy = _ring_centroid(f); lx, ly = STATE_LABEL_XY[nm]
            ax.plot([cx, lx], [cy, ly+0.22], color='0.35', lw=0.35, zorder=7)
            ax.plot([cx], [cy], 'o', ms=1.2, color='0.2', zorder=8)
            _halo(ax.text(lx, ly, STATE_PRETTY.get(nm, nm), fontsize=FS_TICK,
                          ha='center', va='center', color='0.1', zorder=8))
    for f in cm['features']:
        nm = f['properties']['NAME_1']
        if nm in CMR_SAMPLED_REGIONS:
            cx, cy = _ring_centroid(f)
            lab = {'Nord-Ouest': 'North-West', 'Sud-Ouest': 'South-West'}.get(nm, nm)
            lx, ly = CMR_LABEL_XY[nm]
            if nm != 'Centre':
                ax.plot([cx, lx], [cy, ly], color='0.35', lw=0.35, zorder=7)
                ax.plot([cx], [cy], 'o', ms=1.2, color='#7B1E2B', zorder=8)
                _halo(ax.text(lx, ly, lab, fontsize=FS_TICK, ha='center',
                              va='center', color='#7B1E2B', zorder=8))
            else:
                ax.text(lx, ly, lab, fontsize=FS_TICK, ha='center', va='center',
                        color='white', zorder=8)
    for i, (nm, reg, la, lo) in enumerate(SITES, 1):
        ax.plot(lo, la, 'o', ms=5.0, mfc='white', mec='0.1', mew=0.6, zorder=6)
        ax.text(lo, la, str(i), ha='center', va='center', fontsize=4.4,
                color='0.1', zorder=7)
    _halo(ax.text(6.6, 10.6, 'NIGERIA', fontsize=FS_ANN, color='0.3', ha='center',
                  style='italic'), 2.0)
    _halo(ax.text(14.3, 9.6, 'CAMEROON', fontsize=FS_ANN, color='0.3', ha='center',
                  style='italic'), 2.0)
    ax.set_xlim(2.3, 16.5); ax.set_ylim(1.95, 14.9)
    ax.set_title('Sampling locations',
                 fontsize=FS_BASE, loc='left', pad=3, x=0.03)
    ax.set_aspect(1/np.cos(np.radians(7.5)))
    ax.set_xticks([]); ax.set_yticks([]); set_frame(ax, 'none')
def panel_d_keys(ax):
    """Two-row key strip: zones + symbol swatches on top, numbered sites below."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    H = ax.get_window_extent(ax.figure.canvas.get_renderer()).height/ax.figure.dpi*72
    dy = 11.5/H                                     # one key row, in axes fraction
    zl = ['North West', 'North East', 'North Central', 'South West', 'South East',
          'South South']
    y = 1 - 0.55*dy
    ax.text(0.0, y, 'Nigeria geopolitical zone', transform=ax.transAxes,
            fontsize=FS_ANN, weight='bold', va='center', ha='left', color='0.15')
    for c in (0, 1):
        key_block(ax, 0.0 + 0.27*c, y - 1.15*dy,
                  [('sq', ZONE_C[z], None, z) for z in zl[3*c:3*c+3]], dy=dy, sw=0.030)
    key_block(ax, 0.50, y, [('sq', CMR_SAMPLED, None, 'Sampled region, Cameroon'),
                            ('sq', '#FFFFFF', '0.1', 'State shown in b'),
                            ('o', '#FFFFFF', '0.1', 'Sampling site')], dy=dy, sw=0.030)
    y2 = y - 4.5*dy
    ax.text(0.0, y2, 'Cameroon sampling site', transform=ax.transAxes,
            fontsize=FS_ANN, weight='bold', va='center', ha='left', color='0.15')
    for c in range(3):
        for k, (i, (nm, reg, la, lo)) in enumerate(list(enumerate(SITES, 1))[2*c:2*c+2]):
            ax.text(0.0 + 0.255*c, y2 - (1.15+k)*dy, f'{i}  {nm}', fontsize=FS_ANN,
                    va='center', ha='left', color='0.15', transform=ax.transAxes)

# ============================================================ assemble
def build(outfile='Figure1_MPXV_Nigeria_Cameroon.png', figsize=(7.2, 9.45)):
    fig = plt.figure(figsize=figsize, facecolor='w')
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 0.040, 0.125, 1.22],
                          height_ratios=[1.00, 0.44, 1.26],
                          wspace=0.03, hspace=0.22,
                          left=0.012, right=0.988, top=0.972, bottom=0.010)
    axA = fig.add_subplot(gs[:, 0]); axL = fig.add_subplot(gs[:, 1])
    axB = fig.add_subplot(gs[0, 3]); axC = fig.add_subplot(gs[1, 3])
    gsd = gs[2, 3].subgridspec(2, 1, height_ratios=[1, 0.45], hspace=0.01)
    axD = fig.add_subplot(gsd[0]); axDk = fig.add_subplot(gsd[1])
    panel_a(axA, axL); panel_b(axB); panel_c(axC); panel_d(axD); panel_d_keys(axDk)
    for ax, l, dx in [(axA, 'a', 0.0), (axB, 'b', -0.02), (axC, 'c', -0.02), (axD, 'd', -0.02)]:
        ax.text(dx, 1.002, l, transform=ax.transAxes, fontsize=11, weight='bold',
                va='bottom', ha='left')
    fig.savefig(outfile, dpi=400)
    return fig


# ============================================================ script entry
def main(data_dir=None, out_stem='Figure1_MPXV_Nigeria_Cameroon'):
    d = os.path.abspath(data_dir or _HERE)
    d = d.rstrip('/') + '/'
    apply_figure_style(sizes=(FS_BASE, FS_ANN, FS_TICK))
    globals().update(load_data(d))
    globals()['D'] = d
    fig = build(out_stem + '.png', figsize=(7.2, 9.6))
    for ext in ('pdf', 'svg'):
        fig.savefig(f'{out_stem}.{ext}')
    print(f'wrote {out_stem}.png/.pdf/.svg from {d}')
    return fig


if __name__ == '__main__':
    main(*sys.argv[1:3])
