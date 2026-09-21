"""Drawing helpers for the MPXV Nigeria-Cameroon multi-panel figure.

Self-contained: no dependency beyond numpy / pandas / matplotlib.
Run `python build_figure.py` in the folder holding the trees and metadata.
"""
import re, json, collections, numpy as np, pandas as pd, matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle, Polygon as MplPoly, Path, PathPatch, Circle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style
def apply_figure_style(sizes=(9, 8, 7)):
    """Publication rcParams: role-mapped size ladder, outward ticks, embedded fonts."""
    base, secondary, tick = sizes
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.size': base,
        'axes.labelsize': base, 'axes.titlesize': base,
        'legend.fontsize': secondary, 'xtick.labelsize': tick, 'ytick.labelsize': tick,
        'axes.linewidth': 0.6,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'legend.frameon': False,
        'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
        # keep glyphs as editable text, not outlines, in both vector formats
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
        'lines.linewidth': 1.2, 'patch.linewidth': 0.6, 'ps.fonttype': 42})


def set_frame(ax, style='open'):
    """Spine visibility: 'open' (left+bottom), 'boxed' (all), 'none'."""
    show = {'open': (False, False, True, True), 'boxed': (True,)*4,
            'none': (False,)*4}[style]
    for side, vis in zip(('top', 'right', 'bottom', 'left'), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction='out', length=0 if style == 'none' else 3, width=0.6)


# ---------------------------------------------------------------- palettes
# family = hue (A.1 blues, A.2.x oranges, A.3 purple); sublineage = tint within
# family; unassigned lineage (= zoonotic genomes here) in neutral grey.
# A.2.1 takes the remaining Oranges step; 'zoonotic' maps to the palette's
# 'unassigned' grey because those genomes carry no lineage call.
LIN = {'A':'#9C9C9C', 'A.1':'#6BAED6', 'A.2':'#FDD0A2', 'A.2.1':'#FD8D3C',
       'A.2.2':'#E6550D', 'A.2.3':'#8C2D04', 'A.2.4':'#FDAE6B',
       'A.2.5':'#A6761D', 'A.3':'#6A51A3', 'zoonotic':'#E0E0E0'}
LIN_ORDER = ['A', 'A.1', 'A.2', 'A.2.1', 'A.2.4', 'A.2.5', 'A.3',
             'A.2.3', 'A.2.2', 'zoonotic']
LIN_LABEL = {k:('Zoonotic' if k=='zoonotic' else k) for k in LIN}

TIP = {'new':'#1A1A1A', 'cam':'#FF0000', 'pub':'#FFFFFF'}
TIP_EDGE = {'new':'#8A5A10', 'cam':'#8B0000', 'pub':'#8A9199'}
TIP_LABEL = {'new':'Nigeria, this study', 'cam':'Cameroon',
             'pub':'Nigeria, previously published'}

APO_C, NONAPO_C, NOSNP_C = '#7A1220', '#BFC4C9', '#E2E5E8'
BRANCH_C = '#6E757C'

ZONES = {
 'North Central':['Benue','Kogi','Kwara','Nasarawa','Niger','Plateau','FederalCapitalTerritory'],
 'North East':['Adamawa','Bauchi','Borno','Gombe','Taraba','Yobe'],
 'North West':['Jigawa','Kaduna','Kano','Katsina','Kebbi','Sokoto','Zamfara'],
 'South East':['Abia','Anambra','Ebonyi','Enugu','Imo'],
 'South South':['AkwaIbom','Bayelsa','CrossRiver','Delta','Edo','Rivers'],
 'South West':['Ekiti','Lagos','Ogun','Ondo','Osun','Oyo']}
ZONE_C = {'North Central': (0.4980, 0.4327, 0.5229),
          'North East':    (0.8837, 0.7791, 0.1856),
          'North West':    (0.8163, 0.4106, 0.2908),
          'South East':    (0.2810, 0.6396, 0.3951),
          'South South':   '#D27D2D',
          'South West':    (0.4667, 0.7451, 0.8588)}
# states contributing genomes to panel b; the same colours key the panel b tips
STATE_C = {'Edo':'#DAA06D', 'CrossRiver':'#966919', 'AkwaIbom':'#C19A6B',
           'Imo':'#00A36C', 'Abia':'#097969', 'Ondo':'#6495ED'}
STATE_C_PRETTY = {'Edo':'#DAA06D', 'Cross River':'#966919', 'Akwa Ibom':'#C19A6B',
                  'Imo':'#00A36C', 'Abia':'#097969', 'Ondo':'#6495ED'}
CMR_UNSAMPLED, CMR_SAMPLED = '#FFD9D9', '#FF0000'
CMR_SAMPLED_REGIONS = ['Nord-Ouest','Sud-Ouest','Centre']

# ---------------------------------------------------------------- tree parsing
class Nd:
    __slots__ = ("name", "length", "children", "parent", "label")
    def __init__(s):
        s.name = None; s.length = 0.0; s.children = []; s.parent = None; s.label = None

def parse_nexus(path):
    """Minimal NEXUS/Newick reader keeping [&label="..."] internal-node names."""
    txt = open(path).read()
    nw = txt.split("[&R]")[1].strip()
    nw = nw[:nw.rindex(";")+1]
    i, root = 0, Nd()
    cur, nodes = root, [root]
    while i < len(nw):
        ch = nw[i]
        if ch == "(":
            n = Nd(); n.parent = cur; cur.children.append(n); nodes.append(n); cur = n; i += 1
        elif ch == ",":
            n = Nd(); n.parent = cur.parent; cur.parent.children.append(n)
            nodes.append(n); cur = n; i += 1
        elif ch == ")":
            cur = cur.parent; i += 1
        elif ch == ";":
            break
        else:
            m = re.match(r"\s*('(?:[^']|'')*'|[^(),:;\[\]]*)\s*(\[&[^\]]*\])?\s*(:\s*[-\d.eE+]+)?",
                         nw[i:])
            nm, ann, bl = m.group(1), m.group(2), m.group(3)
            if nm: cur.name = nm.strip("'").replace("''", "'")
            if ann:
                lm = re.search(r'label="([^"]+)"', ann)
                if lm: cur.label = lm.group(1)
            if bl: cur.length = float(bl[1:])
            i += m.end() if m.end() > 0 else 1
    return root, nodes

def nid(x):
    """Branch-reconstruction identifier: internal label, or tip name."""
    return x.label if x.label else x.name

_IDPATS = [r"\b([A-Z]{2}\d{6})\b", r"\b(MPOX-\d{2}-\d+)", r"\b(MPX-\d{2}-\d+)",
           r"\b(VSP\d+)", r"\b(TRM\d+)", r"\b(UNTH-\d+)", r"\b(CPHL[-A-Za-z0-9]*)",
           r"\b(NRL[_-][A-Za-z0-9_-]+)"]
def key(s):
    """Sample-ID key so the several tip-naming conventions can be joined."""
    s = str(s).strip()
    for p in _IDPATS:
        m = re.search(p, s, flags=re.I)
        if m: return m.group(1).upper()
    return s.upper()

def is_apobec3(df):
    """APOBEC3 dinucleotide context: G->A at GA, or C->T at TC."""
    return ((df.snp == 'G->A') & (df.dimer == 'GA')) | ((df.snp == 'C->T') & (df.dimer == 'TC'))

def load_data(D):
    """Read trees, reconstructions, metadata and GADM polygons from directory D."""
    root, nodes = parse_nexus(D+"324+NCDC+Nigeria.pruned.tree")
    r2, n2 = parse_nexus(D+"unannotated.tree")
    big = [x.name for x in nodes if not x.children]
    small = [x.name for x in n2 if not x.children]
    sn = pd.read_csv(D+"324+NCDC+Nigeria.tree.branch_snps.reconstruction.csv")
    sn2 = pd.read_csv(D+"unannotated.tree.branch_snps.reconstruction.csv")
    sn['apo'], sn2['apo'] = is_apobec3(sn), is_apobec3(sn2)
    cur = pd.read_csv(D+"MPXV_Nigeria_curated_updated_with_polygon_coords.csv",
                      encoding="mac_roman")
    cur['k'] = cur.Sample.map(key)
    new = pd.read_csv(D+"Nigeria_new_genomes.csv")
    nk = set(new.seqName.map(key))
    cm_ = cur.drop_duplicates('k').set_index('k')
    rows = []
    for t in big:
        k, r = key(t), None
        if key(t) in cm_.index: r = cm_.loc[key(t)]
        rows.append(dict(
            tip=t, key=k, cameroon=('cameroon' in t.lower()), new=(k in nk),
            lineage=(r['lineage'] if r is not None and isinstance(r['lineage'], str) else None),
            zoonotic=(bool(r['animal/h2h'] == 'Zoonotic') if r is not None else True)))
    TM = pd.DataFrame(rows)
    TM['group'] = np.where(TM.cameroon, 'cam', np.where(TM.new, 'new', 'pub'))
    TM['linbar'] = np.where(TM.zoonotic, 'zoonotic', TM.lineage.fillna('zoonotic'))
    ng = json.load(open(D+"gadm41_NGA_1.json"))
    cm = json.load(open(D+"gadm41_CMR_1.json"))
    return dict(root=root, nodes=nodes, r2=r2, n2=n2, big=big, small=small,
                sn=sn, sn2=sn2, TM=TM, nk=nk, ng=ng, cm=cm, curated=cur)

# ---------------------------------------------------------------- tree utils
def nleaf(nd):
    return 1 if not nd.children else sum(nleaf(c) for c in nd.children)

def ladderize(nd, rev=False):
    for ch in nd.children: ladderize(ch, rev)
    if nd.children: nd.children.sort(key=nleaf, reverse=rev)

def layout(root, rev=False):
    """Return (X, Y, tip_order) keyed by id(node); X = cumulative branch length."""
    ladderize(root, rev)
    X, Y, order = {}, {}, []
    def setx(nd, x0):
        X[id(nd)] = x0 + nd.length
        for ch in nd.children: setx(ch, X[id(nd)])
    setx(root, 0.0); X[id(root)] = 0.0
    def sety(nd):
        if not nd.children:
            Y[id(nd)] = len(order); order.append(nd); return Y[id(nd)]
        ys = [sety(c) for c in nd.children]
        Y[id(nd)] = 0.5*(min(ys)+max(ys)); return Y[id(nd)]
    sety(root)
    return X, Y, order

def leaves_of(nd):
    if not nd.children: return [nd]
    out = []
    for ch in nd.children: out += leaves_of(ch)
    return out

def mrca(nodes, names):
    names, best = set(names), None
    for nd in nodes:
        ls = {l.name for l in leaves_of(nd)}
        if names <= ls and (best is None or len(ls) < best[1]): best = (nd, len(ls))
    return best[0]

def branch_segments(nodes, X, Y, colour_of):
    """Orthogonal branch segments (horizontal + vertical connector) with colours."""
    segs, cols, lws = [], [], []
    for nd in nodes:
        p = nd.parent
        if p is None: continue
        c, lw = colour_of(p, nd)
        segs.append([(X[id(p)], Y[id(nd)]), (X[id(nd)], Y[id(nd)])]); cols.append(c); lws.append(lw)
    for nd in nodes:                       # vertical connectors
        if not nd.children: continue
        ys = [Y[id(c)] for c in nd.children]
        segs.append([(X[id(nd)], min(ys)), (X[id(nd)], max(ys))])
        cols.append(NONAPO_C); lws.append(0.5)
    return segs, cols, lws

# ---------------------------------------------------------------- tip labels
STATE_FIX = {'akwaibom':'Akwa Ibom','akwa-ibom':'Akwa Ibom','crossriver':'Cross River','ondo':'Ondo',
             'nord-ouest':'North-West','sud-ouest':'South-West'}
def pretty_tip(name):
    """-> (region, country, year) for panel b labels."""
    yr = re.search(r'(19|20)\d\d', name)
    yr = yr.group(0) if yr else 'NA'
    country = 'Cameroon' if 'cameroon' in name.lower() else 'Nigeria'
    parts = [p for p in re.split(r'[|_]', name) if p]
    drop = {'unpub','mpxv','mpx','nigeria','cameroon'}
    cand = [p for p in parts if p.lower() not in drop
            and not re.match(r'^[A-Z]{2}\d{6}$', p)
            and not re.match(r'^(19|20)\d\d', p)
            and not re.match(r'^(MPOX|MPX|VSP|TRM|UNTH|CPHL|NRL)', p, re.I)
            and not re.match(r'^S\d+$|^L\d+$', p)]
    reg = cand[0] if cand else 'NA'
    k = reg.lower().replace(' ', '')
    if k == 'na': return 'NA', country, yr
    if k in STATE_FIX: reg = STATE_FIX[k]
    else:
        reg = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', reg)
        reg = '-'.join(w.capitalize() for w in reg.split('-'))
        reg = ' '.join(w.capitalize() if w.isupper() or w.islower() else w for w in reg.split())
    return reg, country, yr

# ---------------------------------------------------------------- map utils
def geo_patches(features, facecolour, lw=0.25, ec='white'):
    patches, faces = [], []
    for f in features:
        name = f['properties']['NAME_1']
        fc = facecolour(name)
        for poly in f['geometry']['coordinates']:
            verts, codes = [], []
            for ring in poly:
                r = np.asarray(ring)
                verts.append(r)
                codes += [Path.MOVETO] + [Path.LINETO]*(len(r)-1)
            v = np.vstack(verts)
            patches.append(PathPatch(Path(v, codes)))
            faces.append(fc)
    return PatchCollection(patches, facecolors=faces, edgecolors=ec,
                           linewidths=lw, zorder=1)


def snp_dots(nodes_, X, Y, snp_edges, nid_fn, xscale=1.0, yoff=0.0):
    """Evenly space one dot per reconstructed SNP along each branch.
    snp_edges: {(parent,child): [bool_is_apobec3, ...]}. Returns xs, ys, colours."""
    xs, ys, cs = [], [], []
    for nd in nodes_:
        p = nd.parent
        if p is None: continue
        lst = snp_edges.get((nid_fn(p), nid_fn(nd)))
        if not lst: continue
        lst = sorted(lst, reverse=True)          # APOBEC3 dots proximal, as in source code
        x0, x1 = X[id(p)]*xscale, X[id(nd)]*xscale
        n = len(lst)
        for i, is_apo in enumerate(lst):
            xs.append(x0 + (i+1)*(x1-x0)/(n+1)); ys.append(Y[id(nd)]+yoff)
            cs.append(APO_C if is_apo else NONAPO_C)
    return xs, ys, cs
