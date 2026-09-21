"""Readers and helpers for the MPXV Figure 4 (BEAST time tree + lineage trend).

Self-contained: numpy / pandas / matplotlib only.
"""
import os
import re
import bisect
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- style
def apply_figure_style(sizes=(9, 8, 7)):
    """Publication rcParams: size ladder, outward ticks, editable embedded text."""
    base, secondary, tick = sizes
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.size': base,
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'axes.labelsize': base, 'axes.titlesize': base,
        'legend.fontsize': secondary, 'xtick.labelsize': tick, 'ytick.labelsize': tick,
        'axes.linewidth': 0.6,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'legend.frameon': False,
        'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'ps.fonttype': 42,
        'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
        'lines.linewidth': 1.0, 'patch.linewidth': 0.6})


def set_frame(ax, mode='open'):
    """'open' = left+bottom only; 'none' = no spines."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if mode == 'none':
        for s in ('left', 'bottom'):
            ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- palettes
# origin of each genome in the time tree (as in the circulated version)
GRP_C = {
    'Nigeria, this study': '#2B2B2B', 'Nigeria, previously published': '#BFBFBF',
    'Sierra Leone': '#E8871A', 'Togo': '#7B3294', "Cote d'Ivoire": '#F768A1',
    'Ghana': '#8C510A', 'Guinea': '#1B7837', 'Benin': '#A6D96A',
    'Cameroon': '#D73027', 'Europe': '#92C5DE', 'Asia': '#808000',
    'Americas': '#08306B', 'Africa, other': '#7FCDBB', 'Oceania': '#B15928',
    'Other/Unknown': '#CBC9E2'}
GRP_ORDER = ['Nigeria, this study', 'Nigeria, previously published', 'Sierra Leone',
             'Togo', "Cote d'Ivoire", 'Ghana', 'Guinea', 'Benin', 'Cameroon',
             'Europe', 'Asia', 'Americas', 'Africa, other', 'Oceania',
             'Other/Unknown']

# lineage palette shared with Figure 1 (family = hue, sublineage = tint);
# the exported G/H lineages continue the scheme in distinct hues.
LIN_C = {'A': '#9C9C9C', 'A.1': '#6BAED6', 'A.1.1': '#9ECAE1', 'A.2': '#FDD0A2',
         'A.2.1': '#FD8D3C', 'A.2.2': '#E6550D', 'A.2.3': '#8C2D04',
         'A.2.4': '#FDAE6B', 'A.2.5': '#A6761D', 'A.3': '#6A51A3',
         'B.1': '#238B8B', 'B.1.2': '#66C2C2', 'B.1.23': '#A5DBDB',
         'F.4': '#B2ABD2', 'G.1': '#E8871A', 'G.2': '#7B3294',
         'H.1': '#1B7837', 'H.2': '#66BD63', 'zoonotic': '#E0E0E0'}

# focal clades whose posterior tMRCAs are drawn beneath the tree
FOCAL_C = {'G1': '#E8871A', 'G2': '#7B3294', 'CIV': '#F768A1'}
FOCAL_LABEL = {'G1': 'G.1, Sierra Leone', 'G2': 'G.2, Togo',
               'CIV': "Cote d'Ivoire 2025"}


# ------------------------------------------------------------------ tree model
class Node:
    __slots__ = ('name', 'length', 'children', 'parent', 'ann', 'depth', 'height',
                 'x', 'y')

    def __init__(self):
        self.name = None
        self.length = 0.0
        self.children = []
        self.parent = None
        self.ann = {}
        self.depth = 0.0
        self.height = 0.0
        self.x = 0.0
        self.y = 0.0


def _split_top(s, sep=','):
    """Split on sep at brace depth 0 (BEAST annotations nest {} but not [])."""
    out, buf, d = [], [], 0
    for ch in s:
        if ch in '{[':
            d += 1
        elif ch in '}]':
            d -= 1
        if ch == sep and d == 0:
            out.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    out.append(''.join(buf))
    return out


def parse_annotation(block):
    """'[&a=1,b={2,3}]' -> {'a': 1.0, 'b': [2.0, 3.0]}; non-numeric kept as str."""
    body = block[block.index('&') + 1:].rstrip(']')
    ann = {}
    for part in _split_top(body):
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        v = v.strip()
        if v.startswith('{'):
            vals = [x.strip() for x in _split_top(v[1:-1])]
            try:
                ann[k.strip()] = [float(x) for x in vals]
            except ValueError:
                ann[k.strip()] = vals
        else:
            try:
                ann[k.strip()] = float(v)
            except ValueError:
                ann[k.strip()] = v
    return ann


_TOK = re.compile(r"""\s*(?:
    (?P<open>\()|(?P<close>\))|(?P<comma>,)|(?P<colon>:)|(?P<semi>;)|
    (?P<ann>\[[^\]]*\])|
    (?P<label>'(?:[^']|'')*'|[^(),:;\[\]\s]+)
)""", re.X)


def parse_newick(s, keep_ann=True):
    """Parse one Newick/BEAST tree string. Returns (root, nodes)."""
    root = Node()
    nodes = [root]
    cur = root
    i, n = 0, len(s)
    expect_len = False
    while i < n:
        m = _TOK.match(s, i)
        if not m:
            i += 1
            continue
        i = m.end()
        g = m.lastgroup
        if g == 'open':
            ch = Node(); ch.parent = cur; cur.children.append(ch); nodes.append(ch); cur = ch
        elif g == 'comma':
            ch = Node(); ch.parent = cur.parent; cur.parent.children.append(ch)
            nodes.append(ch); cur = ch
        elif g == 'close':
            cur = cur.parent
        elif g == 'semi':
            break
        elif g == 'colon':
            expect_len = True
        elif g == 'ann':
            if keep_ann:
                cur.ann = parse_annotation(m.group('ann'))
        elif g == 'label':
            lab = m.group('label')
            if expect_len:
                try:
                    cur.length = float(lab)
                except ValueError:
                    pass
                expect_len = False
            else:
                cur.name = lab.strip("'").replace("''", "'")
    return root, nodes


def read_nexus(path, strip_ann=False, max_trees=None):
    """Yield (tree_name, newick_string) and return the Translate map.

    Returns (translate, [(name, newick), ...]).  strip_ann removes [&...] blocks,
    which makes parsing the posterior tree file several times faster.
    """
    translate, trees = {}, []
    in_trans = False
    with open(path) as fh:
        for line in fh:
            ls = line.strip()
            if ls.lower().startswith('translate'):
                in_trans = True
                continue
            if in_trans:
                m = re.match(r"^(\d+)\s+'?(.+?)'?,?$", ls)
                if m:
                    translate[m.group(1)] = m.group(2)
                if ls.endswith(';'):
                    in_trans = False
                continue
            if ls.lower().startswith('tree '):
                name = ls.split()[1]
                nwk = ls[ls.index('=') + 1:].strip()
                if nwk.startswith('[&R]'):
                    nwk = nwk[4:].strip()
                if strip_ann:
                    nwk = re.sub(r'\[[^\]]*\]', '', nwk)
                trees.append((name, nwk))
                if max_trees and len(trees) >= max_trees:
                    break
    return translate, trees


# ------------------------------------------------------------------ tree maths
def leaves(nd):
    """Tip nodes below nd (iterative: posterior trees can be deep ladders)."""
    out, stack = [], [nd]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            out.append(n)
    return out


def set_depths(root):
    """Cumulative branch length from the root; returns the maximum over tips."""
    stack = [root]
    root.depth = 0.0
    dmax = 0.0
    while stack:
        nd = stack.pop()
        for ch in nd.children:
            ch.depth = nd.depth + ch.length
            stack.append(ch)
        if not nd.children:
            dmax = max(dmax, nd.depth)
    return dmax


def set_dates(root, most_recent):
    """Height above the most recent tip -> calendar date on every node."""
    dmax = set_depths(root)
    stack = [root]
    while stack:
        nd = stack.pop()
        nd.height = dmax - nd.depth
        nd.x = most_recent - nd.height
        stack.extend(nd.children)
    return dmax


def mrca(root, names):
    """Smallest node whose leaf set contains all of names."""
    want = set(names)
    best = None
    stack = [root]
    while stack:
        nd = stack.pop()
        ls = {l.name for l in leaves(nd)}
        if want <= ls:
            best = nd
            stack = list(nd.children)
    return best


def is_monophyletic(root, names):
    nd = mrca(root, names)
    return nd is not None and {l.name for l in leaves(nd)} == set(names)


# ------------------------------------------------------------------ dates
def calibrate(root, tip_dates, day_only=None):
    """Offset mapping node height to calendar date, fitted on fully-dated tips.

    Returns C such that date = C - height. Fitting on all day-resolution tips
    instead of anchoring on one tip absorbs whatever decimal-date convention the
    BEAST run used, and leaves tips whose ages BEAST sampled out of the fit.
    """
    dmax = set_depths(root)
    est = []
    for nd in (l for l in leaves(root)):
        if nd.name not in tip_dates or tip_dates[nd.name] is None:
            continue
        if day_only is not None and nd.name not in day_only:
            continue
        est.append(tip_dates[nd.name] + (dmax - nd.depth))
    C = float(np.median(est))
    resid = np.abs(np.array(est) - C) * 365.25
    return C, resid


def apply_dates(root, C):
    dmax = set_depths(root)
    for nd in (n for n in _walk(root)):
        nd.height = dmax - nd.depth
        nd.x = C - nd.height


def _walk(root):
    stack = [root]
    while stack:
        nd = stack.pop()
        yield nd
        stack.extend(nd.children)


def mrca_fast(root, names, parents=None):
    """MRCA of `names` in O(nodes) via a post-order count. Returns (node, parent)."""
    want = set(names)
    order, stack, par = [], [root], {id(root): None}
    while stack:
        nd = stack.pop()
        order.append(nd)
        for ch in nd.children:
            par[id(ch)] = nd
            stack.append(ch)
    cnt = {}
    for nd in reversed(order):                       # children before parents
        c = sum(cnt[id(ch)] for ch in nd.children)
        if not nd.children and nd.name in want:
            c = 1
        cnt[id(nd)] = c
    n = len(want)
    if cnt[id(root)] < n:
        return None, None
    nd = root
    while True:
        nxt = next((ch for ch in nd.children if cnt[id(ch)] == n), None)
        if nxt is None:
            return nd, par[id(nd)]
        nd = nxt


def posterior_node_dates(path, sets, tip_dates, day_only, burnin=0.1, progress=None):
    """MRCA and stem dates of each taxon set across a posterior tree file.

    Every tree is calibrated on its own fully-dated tips (see calibrate), so
    tips whose ages BEAST sampled cannot shift the time scale. Returns one row
    per post-burnin tree.
    """
    translate, trees = read_nexus(path, strip_ann=True)
    inv = {v: k for k, v in translate.items()}
    num_sets = {k: {inv[n] for n in v if n in inv} for k, v in sets.items()}
    num_date = {inv[n]: d for n, d in tip_dates.items() if n in inv and d is not None}
    num_day = {inv[n] for n in day_only if n in inv}
    start = int(len(trees) * burnin)
    rows = []
    for state, (nm, nwk) in enumerate(trees):
        if state < start:
            continue
        rt, _ = parse_newick(nwk, keep_ann=False)
        C, _ = calibrate(rt, num_date, day_only=num_day)
        dmax = set_depths(rt)
        rec = {'state': nm}
        for lab, want in num_sets.items():
            node, parent = mrca_fast(rt, want)
            if node is None:
                rec[lab] = rec[lab + '_stem'] = np.nan
                rec[lab + '_mono'] = False
                continue
            rec[lab] = C - (dmax - node.depth)
            rec[lab + '_stem'] = (C - (dmax - parent.depth)
                                  if parent is not None else np.nan)
            rec[lab + '_mono'] = len(leaves(node)) == len(want)
        rows.append(rec)
        if progress and len(rows) % progress == 0:
            print(f'  {len(rows)} trees', flush=True)
    return pd.DataFrame(rows)


def hpd(x, mass=0.95):
    """Highest posterior density interval of a 1-D sample."""
    x = np.sort(np.asarray(x)[~np.isnan(x)])
    if x.size == 0:
        return (np.nan, np.nan)
    k = max(1, int(np.floor(mass * x.size)))
    if k >= x.size:
        return (x[0], x[-1])
    w = x[k:] - x[:x.size - k]
    i = int(np.argmin(w))
    return (x[i], x[i + k])


def gaussian_kde_1d(x, grid, bw=None):
    """Gaussian KDE with Silverman bandwidth; no scipy dependency."""
    x = np.asarray(x)[~np.isnan(np.asarray(x))]
    if x.size < 2:
        return np.zeros_like(grid)
    if bw is None:
        s = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        a = min(s, iqr / 1.349) if iqr > 0 else s
        bw = 0.9 * a * x.size ** (-0.2) if a > 0 else 0.01
    bw = max(bw, 1e-4)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))


_CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def decimal_date(y, m=None, d=None):
    """Calendar date -> decimal year using real month lengths (mid-day convention).

    Partial dates fall at the middle of the stated month or year, matching how
    BEAST treats them before sampling their ages.
    """
    if m is None:
        return y + 0.5
    if d is None:
        d = 15
    return y + (_CUM[m - 1] + d - 0.5) / 365.25


def date_from_label(name):
    """Trailing |YYYY[-MM[-DD]] in a taxon label -> decimal year, or None."""
    last = str(name).strip().split('|')[-1]
    m = re.fullmatch(r'(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?', last)
    if not m:
        return None
    y = int(m.group(1))
    if not 1950 <= y <= 2030:
        return None
    mo = int(m.group(2)) if m.group(2) else None
    dd = int(m.group(3)) if m.group(3) else None
    return decimal_date(y, mo, dd)


def to_calendar(dec):
    """Decimal year -> 'YYYY-MM-DD' (proleptic, ignores leap days)."""
    y = int(np.floor(dec))
    doy = (dec - y) * 365.25 + 0.5
    mi = max(0, int(bisect.bisect_right(_CUM, doy)) - 1)
    day = int(round(doy - _CUM[mi])) + 1
    if day > [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mi]:
        mi, day = min(11, mi + 1), 1
    return f'{y}-{mi + 1:02d}-{day:02d}'


# ------------------------------------------------------------------ data load
NG_STATES = {'ABIA', 'ADAMAWA', 'AKWAIBOM', 'ANAMBRA', 'BAUCHI', 'BAYELSA', 'BENUE',
             'BORNO', 'CROSSRIVER', 'DELTA', 'EBONYI', 'EDO', 'EKITI', 'ENUGU', 'FCT',
             'GOMBE', 'IMO', 'JIGAWA', 'KADUNA', 'KANO', 'KATSINA', 'KEBBI', 'KOGI',
             'KWARA', 'LAGOS', 'NASARAWA', 'NIGER', 'OGUN', 'ONDO', 'OSUN', 'OYO',
             'PLATEAU', 'RIVERS', 'SOKOTO', 'TARABA', 'YOBE', 'ZAMFARA', 'NIGERIA'}

# country token -> legend group, tested in order (first match wins)
COUNTRY_TOK = [
    ('Sierra[_ ]?Leone', 'Sierra Leone'), ('Togo', 'Togo'),
    ("Cote[_ ]?d.?Ivoire|Ivoire|Ivory", "Cote d'Ivoire"), ('Ghana', 'Ghana'),
    ('Guinea|Conakry', 'Guinea'), ('Benin', 'Benin'), ('Cameroon', 'Cameroon'),
    ('DRC|Congo', 'Africa, other'),
    ('Zambia|Kenya|Uganda|Senegal|Morocco|South[_ ]?Africa|Malawi|Mozambique|'
     'Ethiopia|Tanzania|Rwanda|Burundi|Sudan|Egypt|Liberia|Gabon', 'Africa, other'),
    ('USA|Canada|Brazil|Mexico|Peru|Colombia|Chile|Argentina|Ecuador|Guatemala',
     'Americas'),
    ('Germany|Netherlands|UK|England|Scotland|Wales|Ireland|Spain|France|Portugal|'
     'Italy|Belgium|Sweden|Switzerland|Austria|Denmark|Norway|Finland|Poland|Czech|'
     'Greece|Slovenia|Slovakia|Hungary|Croatia|Russia|Malta|Luxembourg', 'Europe'),
    ('Israel|Singapore|Thailand|India|Japan|China|Korea|UAE|Pakistan|Nepal|Jordan|'
     'Vietnam|Philippines|Indonesia|Taiwan|Malaysia|Qatar|Oman|Lebanon|Turkey|'
     'Bangladesh|SriLanka', 'Asia'),
    ('Australia|NewZealand|New[_ ]?Zealand', 'Oceania')]


def origin_group(name):
    """Legend group for a taxon label: Nigerian state names resolve to Nigeria."""
    n = str(name)
    for tok, g in COUNTRY_TOK:
        if re.search(tok, n, re.I):
            return g
    parts = {p.upper().replace(' ', '').replace('-', '') for p in re.split(r'[|_]', n)}
    if parts & NG_STATES:
        return 'Nigeria'
    return 'Other/Unknown'


def sample_key(s):
    """Shared id across metadata files and tree labels."""
    m = re.search(r'\b((?:NRL[-_])?(?:CPHL[-_])?(?:MPOX|MPX)[-_]\d+|UNTH-\d+|VSP\d+'
                  r'|TRM\d+|[A-Z]{2}\d{6})\b', str(s), re.I)
    return m.group(1).upper() if m else str(s).upper()


def sample_year(name):
    """Collection year from a trailing date or from the id (MPOX-24-..)."""
    n = str(name)
    m = re.search(r'\|(\d{4})-\d{2}-\d{2}$|\|(\d{4})-\d{2}$|\|(\d{4})$', n)
    if m:
        return int(next(g for g in m.groups() if g))
    m = re.match(r'^(?:NRL[-_])?(?:CPHL[-_])?(?:MPOX|MPX)[-_](\d{2})[-_]', n, re.I)
    if m:
        return 2000 + int(m.group(1))
    return None


def date_format(name):
    last = str(name).split('|')[-1]
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', last):
        return 'day'
    if re.fullmatch(r'\d{4}-\d{2}', last):
        return 'month'
    return 'year' if re.fullmatch(r'\d{4}', last) else 'none'


def maximal_clusters(root, names):
    """Maximal nodes whose leaf set lies entirely inside names, largest first."""
    names, out = set(names), []

    def rec(nd):
        ls = {l.name for l in leaves(nd)}
        if ls <= names:
            out.append(nd)
            return
        for c in nd.children:
            rec(c)
    rec(root)
    return sorted(out, key=lambda n: -len(leaves(n)))


def load_data4(d, burnin=0.1, posterior=True):
    """Read every Figure 4 input and return the objects the panels need."""
    d = d.rstrip('/') + '/'
    tree_file = next(f for f in os.listdir(d) if f.endswith('.tree'))
    translate, trees = read_nexus(d + tree_file)
    root, nodes = parse_newick(trees[0][1])
    for nd in nodes:
        if not nd.children and nd.name in translate:
            nd.name = translate[nd.name]
    tips = [nd for nd in nodes if not nd.children]

    tip_date = {nd.name: date_from_label(nd.name) for nd in tips}
    missing = [k for k, v in tip_date.items() if v is None]
    if missing:
        raise ValueError(f'{len(missing)} tips carry no parseable date, e.g. {missing[:2]}')
    day_only = {nd.name for nd in tips if date_format(nd.name) == 'day'}
    C, resid = calibrate(root, tip_date, day_only=day_only)
    apply_dates(root, C)

    nx = pd.read_csv(d + 'nextclade.tsv', sep='\t')
    lin = dict(zip(nx.seqName, nx.lineage))
    new_ids = {sample_key(s) for s in pd.read_csv(d + 'Nigeria_new_genomes.csv').seqName}

    meta = pd.DataFrame({'tip': [nd.name for nd in tips]})
    meta['date'] = meta.tip.map(tip_date)
    meta['lineage'] = meta.tip.map(lin)
    meta['origin'] = meta.tip.map(origin_group)
    meta['is_new'] = meta.tip.map(lambda t: sample_key(t) in new_ids)
    meta['group'] = np.where(meta.origin != 'Nigeria', meta.origin,
                             np.where(meta.is_new, 'Nigeria, this study',
                                      'Nigeria, previously published'))

    # focal clades: the two exported lineages, and the largest Ivorian cluster
    sets = {}
    for key, lineage in (('G1', 'G.1'), ('G2', 'G.2')):
        names = set(meta.tip[meta.lineage == lineage])
        nd, _ = mrca_fast(root, names)
        sets[key] = {l.name for l in leaves(nd)}
    civ = set(meta.tip[meta.origin == "Cote d'Ivoire"])
    sets['CIV'] = {l.name for l in leaves(maximal_clusters(root, civ)[0])}

    post = None
    if posterior and os.path.exists(d + 'beast.trees'):
        post = posterior_node_dates(d + 'beast.trees', sets, tip_date, day_only,
                                    burnin=burnin)

    cur = pd.read_csv(d + 'MPXV_Nigeria_curated_updated_with_polygon_coords.csv',
                      encoding='mac_roman')
    cur = cur[cur.country == 'Nigeria'].copy()
    cur['year'] = cur.Sample.map(sample_year)
    cur['human'] = cur['animal/h2h'].ne('Zoonotic')
    cur['lineage_plot'] = np.where(cur.human, cur.lineage.fillna('zoonotic'), 'zoonotic')
    # exact collection dates where the same sample appears in the tree or in
    # nextclade; otherwise the year encoded in the sample id, at mid-year
    exact = {sample_key(t): dd for t, dd in zip(meta.tip, meta.date)}
    for s in nx.seqName:
        dd = date_from_label(s)
        if dd is not None:
            exact.setdefault(sample_key(s), dd)
    cur['k'] = cur.Sample.map(sample_key)
    cur['t_exact'] = cur.k.map(exact.get)
    cur['t'] = cur.t_exact.where(cur.t_exact.notna(), cur.year + 0.5)

    return dict(root=root, nodes=nodes, tips=tips, meta=meta, C=C, calib_resid=resid,
                sets=sets, post=post, curated=cur, translate=translate)
