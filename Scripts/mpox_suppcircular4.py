#!/usr/bin/env python3
"""
MPOX circular divergence tree + full-world map (Dudas B.1.620 layout).

Faithful to the Dudas B.1.620 layout:
  * Full-world map (Robinson, set_global) in the centre so AMERICA and ASIA show.
  * Tips sit near the centre OVER the map (fixed INWARD, as Gytis), so the tree
    branches radiate outward and cross the scale rings (not separated from them).
  * Scale rings span the WHOLE tree (ticks across 0..treeHeight).
  * Legends arranged in the corners around the tree (Africa top-left, rest top-right).
  * Distinct map palette (warm land / pale-teal sea), not Gytis's neutral grey.
  * Every visible country gets a map marker; tip->map LINES only for African
    neighbours (LINE_ONLY_AFRICA) so the centre is not a spaghetti of rays.
  * APOBEC3 / non-APOBEC3 SNP dots dropped; Nigeria tips carry NO lines (anchor dot).
  * Nigeria: NEW seqs = deep grey, older = light grey. Continent-lumped palette.
  * Sierra Leone G.1 thinned to SL_KEEP reps, every other (unique-country) tip kept.

Requires: baltic, cartopy, numpy, pandas, matplotlib.
"""
import os
from collections import Counter
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import baltic as bt

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'normal', 'font.size': 25,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
})

# ============================== CONFIG (edit paths) ==============================
BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
TREEFILE      = os.path.join(BASE, "global.tree")
HIGHLIGHT_CSV = os.path.join(BASE, "Nigeria_new_genomes.csv")
COLLAPSE_CSV  = None                      # collapse+tree.csv absent; G.1 derived from Nextclade
NEXTCLADE_TSV = os.path.join(BASE, "nextclade_lineages.tsv")
OUTFILE       = os.path.join(BASE, "panelB_baseline")

# cartopy's default cache is not writable here; keep basemap shapefiles beside the script
import cartopy as _cartopy
_CARTOPY_DIR = os.environ.get("CARTOPY_DATA_DIR",
                              os.path.join(os.path.dirname(BASE), "cartopy_data"))
_cartopy.config["data_dir"] = _CARTOPY_DIR
_cartopy.config["pre_existing_data_dir"] = _CARTOPY_DIR

MPOX_ALN_LENGTH = 197209

# Nigeria greys (set NIGERIA_GREY=False to use steelblue from AFRICA_COLOURS instead)
NIGERIA_GREY     = True
NIGERIA_NEW_GREY = '#404040'
NIGERIA_OLD_GREY = '#BDBDBD'
DRAW_NIGERIA_ANCHOR = True          # one grey Nigeria dot on the map (no lines)

# zoom / layout knobs (Gytis geometry: tips sit near centre, OVER the map)
MAP_WIDTH   = 0.40                 # central world-map size (fraction of figure)
MAP_CLON    = 10                   # map central longitude (Africa-centred)
GAP_FRAC    = 0.85                 # tips ring at GAP_FRAC/1.1 of radius; map sits in the central hole
                                   #   (Gytis = -34/treeHeight ≈ 0.85). Bigger -> tips further out, more gap
SL_KEEP     = 20                   # Sierra Leone G.1 thinned to this many; unique countries retained
LINE_ONLY_AFRICA = False           # False -> Europe/Asia/Americas connectors drawn too
LINE_PER_COUNTRY = True            # ONE connector per country (Gytis keeps the centre clean)


# ─── Text and marker sizes ───────────────────────────────────────────────
# Requested: scale text = 25, legend text = 25; other tips bigger, but still
# smaller than the new Nigerian sequences.
SCALE_TEXT_SIZE = 45
LEGEND_TEXT_SIZE = 45
BRANCH_LINEWIDTH = 3.2            # circular-tree branches; raised so they read clearly
BRANCH_HALO_MULT = 1.45            # white casing behind each branch (was 3x, too washed out)
NODE_ARC_LINEWIDTH = 3.2          # the radial arcs joining sister clades; matched to the branches
SCALE_BAR_LINEWIDTH = 1.0         # mutation scale bar and its ticks
RING_LINEWIDTH = 0.8              # dashed mutation rings crossing the tree
MAP_COAST_LINEWIDTH = 0.4         # land edge on the central map
MAP_BORDER_LINEWIDTH = 0.5        # country borders on the central map
TIP_SIZE = 115                    # normal/non-highlighted tips; was 80
AFRICA_TIP_SIZE = 125             # African non-highlighted tips; still < new Nigerian tips
NEW_NIGERIAN_TIP_SIZE = 220       # highlighted/new Nigerian sequences; biggest tips
TIP_HALO_FACTOR = 1.50            # black halo around important tips

MAP_MARKER_SIZE = 145
MAP_MARKER_HALO_SIZE = 230
NIGERIA_MAP_MARKER_SIZE = 220
NIGERIA_MAP_HALO_SIZE = 340

# ─── Clustered tip-to-map connectors ─────────────────────────────────────
# Instead of many tip-to-map rays, nearby tips first converge into one small
# external point, then only ONE line from that point links to the map.
CLUSTER_CONNECTORS = True
CONNECT_NIGERIA_TO_MAP = True     # set False if you want only the Nigeria map dot, no Nigeria lines
CLUSTER_ANGLE_GAP_DEGREES = 9.0   # tips within this angular distance are treated as one cluster
CONNECTOR_RADIAL_PAD = 0.055      # distance of convergence point outside the tree rim
CONNECTOR_CLUSTER_POINT_SIZE = 100
CONNECTOR_LINEWIDTH = 1.25
CONNECTOR_ALPHA = 0.45

# ─── Creative lineage arcs outside the final mutation ring ──────────────
# `unassigned` in Nextclade is treated as the zoonotic group.
# This version places lineage arcs JUST OUTSIDE the outermost mutation-scale
# circle. The labels are outside the arcs, so nothing goes into the map/centre.
SHOW_LINEAGE_ARCS = True           # draw lineage arcs/ribbons outside the scale rings
SHOW_LINEAGE_LIST = False          # no separate lineage list; labels sit on the outer arcs
LINEAGE_TEXT_SIZE = 40             # requested lineage label font size
LINEAGE_LIST_FONT_SIZE = 40
LINEAGE_LIST_NCOL = 3
LINEAGE_LIST_LOC = 'lower center'
LINEAGE_LIST_BBOX = (0.50, 0.015)
LINEAGE_LIST_FRAMEON = False
LINEAGE_LIST_SHOW_COUNTS = False
LINEAGE_LIST_TITLE = 'Lineages'

# These values are distances in the polar plot coordinate system.
# The arc radius is computed from the LAST mutation-scale ring, then these
# small pads move lineage tracks just outside that final ring.
LINEAGE_ARC_LINEWIDTH = 6.0
LINEAGE_ARC_ALPHA = 0.92
LINEAGE_ARC_MIN_TIPS = 1           # show every detected lineage, including small lineages
LINEAGE_ARC_GAP_DEGREES = 18.0     # split same lineage if it appears in separated tree regions
LINEAGE_MAX_SEGMENTS_PER_LINEAGE = 6
LINEAGE_ARC_RADIUS_PAD = 0.045     # distance OUTSIDE the final mutation ring
LINEAGE_ARC_TRACK_SPACING = 0.034  # separates multiple lineage tracks outside the ring
LINEAGE_LABEL_RADIAL_PAD = 0.022   # label baseline outside each lineage arc
LINEAGE_ARC_ZORDER = 30            # visible outside the tree
LINEAGE_LABEL_ZORDER = 31
LINEAGE_LABEL_HALO_WIDTH = 3.2
LINEAGE_LABEL_MODE = 'tangent'   # 'horizontal' is most readable; use 'tangent' if preferred
LINEAGE_LABEL_ONLY_LARGEST_SEGMENT = True
LINEAGE_LABEL_BOX = False            # white rounded background keeps text readable
LINEAGE_LEADER_LINE = False          # small line from arc to horizontal label

LINEAGE_LABELS_CURVED = False       # draw labels as circular text following the arc
LINEAGE_ARC_LABEL_USE_SHORT_CODE = True   # use G.1, H.1, A.2.2 etc. on the arc
LINEAGE_LABEL_MIN_SPAN_DEGREES = 2.0      # short codes remain readable on small arcs
LINEAGE_LABEL_STAGGER = False             # superseded by the measured de-collision below
# Measured label de-collision: labels are collected, their true angular widths
# are measured from the renderer, and crowded ones are fanned apart in angle
# and stepped onto alternating radial tracks, each keeping a leader line back
# to its own arc. This is what separates B.1 / B.1.2 / B.1.23 / A.1.1 / F.4,
# which occupy almost the same angle at the bottom of the circle.
LINEAGE_LABEL_DECOLLIDE   = True
LINEAGE_LABEL_GAP_PAD     = 1.14   # required clearance as a multiple of label width
LINEAGE_LABEL_TRACK_STEP  = 0.072  # radial step between alternating label tracks
LINEAGE_LABEL_LEADER_LW   = 1.4
LINEAGE_LABEL_LEADER_ALPHA = 0.85
LINEAGE_LABEL_LADDER_MIN  = 3      # clusters this size or larger go on a radial ladder
LINEAGE_LABEL_LADDER_PAD  = 1.30   # ladder rung spacing, in measured text lines
LINEAGE_LABEL_LADDER_START = 0.085 # first rung starts clear of the neighbouring arcs
LINEAGE_LABEL_CANVAS_MARGIN = 0.045
LINEAGE_LABEL_LADDER_FONT = 0.80   # minor lineages on the ladder, relative size  # room reserved for the ladder, each side

LINEAGE_LABEL_EVERY_SEGMENT = False # False = one label per lineage, on the largest segment

# Keep the important lineages first, then add every other detected lineage automatically.
LINEAGE_LABEL_ORDER = [
    'G.1', 'A.2.2', 'A.2.3', 'A', 'A.2.1', 'A.1', 'A.2', 'Zoonotic',
    'H.1', 'A.2.4', 'A.3', 'H.2', 'A.2.5'
]
LINEAGE_LABEL_NAMES = {
    'G.1':      'G.1 – Sierra Leone outbreak',
    'H.1':      'H.1 – India/Asia-Europe',
    'H.2':      'H.2 – India/Pakistan-linked',
    'A.2.2':    'A.2.2 – recent Nigerian backbone',
    'A.2.3':    'A.2.3 – Nigeria-dominated',
    'A.2.1':    'A.2.1 – older international backbone',
    'A.2':      'A.2',
    'A.2.4':    'A.2.4',
    'A.2.5':    'A.2.5',
    'A.1':      'A.1',
    'A.3':      'A.3',
    'A':        'A',
    'Zoonotic': 'Zoonotic',
}
LINEAGE_LABEL_COLOURS = {
    'G.1':      '#E07B00',
    'H.1':      '#009E73',
    'H.2':      '#7B2D8B',
    'A.2.2':    '#4D4D4D',
    'A.2.3':    '#0072B2',
    'A.2.1':    '#CC79A7',
    'A.2':      '#56B4E9',
    'A.2.4':    '#D55E00',
    'A.2.5':    '#B8960B',
    'A.1':      '#999999',
    'A.3':      '#8B4513',
    'A':        '#6B6B6B',
    'Zoonotic': '#D73027',
}

# Small x/y nudges in data coordinates for crowded labels. Usually leave at zero;
# use these only if two labels overlap in the final plot.
LINEAGE_LABEL_NUDGE = {
    'G.1': (0.000, 0.020),
    'H.1': (0.000, 0.000),
    'H.2': (0.000, -0.015),
    'A.2.2': (0.000, 0.030),
    'A.2.3': (0.000, 0.000),
}

def set_size_scale(scale, fonts=None):
    """Rescale every marker/linewidth constant for a smaller physical panel.

    The panel's geometry was specified for a ~28 in canvas. Marker `s` values are
    areas, so they scale with scale**2; linewidths scale linearly. Font sizes are
    set from `fonts` instead of scaled, so the figure keeps a deliberate type
    ladder rather than whatever the scale factor lands on.
    """
    g = globals()
    for name in ('TIP_SIZE', 'AFRICA_TIP_SIZE', 'NEW_NIGERIAN_TIP_SIZE',
                 'MAP_MARKER_SIZE', 'MAP_MARKER_HALO_SIZE',
                 'NIGERIA_MAP_MARKER_SIZE', 'NIGERIA_MAP_HALO_SIZE',
                 'CONNECTOR_CLUSTER_POINT_SIZE'):
        g[name] = g[name] * scale ** 2
    for name in ('BRANCH_LINEWIDTH', 'NODE_ARC_LINEWIDTH', 'CONNECTOR_LINEWIDTH',
                 'LINEAGE_ARC_LINEWIDTH', 'LINEAGE_LABEL_LEADER_LW',
                 'SCALE_BAR_LINEWIDTH', 'RING_LINEWIDTH',
                 'MAP_COAST_LINEWIDTH', 'MAP_BORDER_LINEWIDTH'):
        g[name] = g[name] * scale
    if fonts:
        for name, val in fonts.items():
            g[name] = val
    return {k: g[k] for k in ('TIP_SIZE', 'BRANCH_LINEWIDTH', 'LINEAGE_ARC_LINEWIDTH',
                              'SCALE_TEXT_SIZE', 'LEGEND_TEXT_SIZE', 'LINEAGE_TEXT_SIZE')}


# ─── Country / continent palette ────────────────────────────────────────
COUNTRY_DISPLAY = {
    'Cote_dIvoire': "C\u00f4te d'Ivoire", 'Sierra_Leone': 'Sierra Leone',
}
AFRICA_COLOURS = {
    "Nigeria": "steelblue", "Cameroon": "red", "Sierra_Leone": "#E07B00",
    "Togo": "#7B2D8B", "Guinea": "#2CA02C", "Ghana": "#8B4513",
    "Cote_dIvoire": "#FF69B4", "Zambia": "#FAD5A5",
}
CONTINENT_COLOURS = {
    "Europe": "#89CFF0", "Asia": "#808000", "Americas": "#00008B", "Other": "#BDB5D5",
}
EUROPE_COUNTRIES = {
    "UK", "United_Kingdom", "United Kingdom", "Germany", "Ireland", "Netherlands",
    "Italy", "Portugal", "Austria", "France", "Spain", "Belgium", "Switzerland",
    "Denmark", "Sweden", "Norway", "Finland", "Poland", "Czech", "Czech_Republic",
}
ASIA_COUNTRIES = {
    "India", "Australia", "New_Zealand", "South_Korea", "South Korea", "Japan",
    "China", "Vietnam", "Thailand", "Singapore", "Taiwan", "Israel", "Oman",
    "UAE", "Qatar", "Saudi_Arabia", "Saudi Arabia",
}
AMERICAS_COUNTRIES = {
    "USA", "Canada", "Brazil", "Argentina", "Mexico", "Colombia", "Peru",
}
_ALIASES = {
    "Sierra Leone": "Sierra_Leone", "Cote dIvoire": "Cote_dIvoire",
    "United_Kingdom": "UK", "United Kingdom": "UK", "Cameroun": "Cameroon",
    "South Korea": "South_Korea", "SouthKorea": "South_Korea",
    "Saudi Arabia": "Saudi_Arabia", "SaudiArabia": "Saudi_Arabia",
}
ALL_COUNTRIES = set(AFRICA_COLOURS) | EUROPE_COUNTRIES | ASIA_COUNTRIES | AMERICAS_COUNTRIES


# lat, lon per location (Nigeria anchor included)
COORDINATES = {
    'Nigeria': (9.082, 8.675), 'Cameroon': (3.848, 11.502), 'Sierra_Leone': (8.461, -11.780),
    'Togo': (8.620, 0.825), 'Guinea': (11.0, -15.0), 'Ghana': (7.947, -1.023),
    'Cote_dIvoire': (7.540, -5.547), 'Zambia': (-15.39, 28.32),
    'USA': (38.895, -77.037), 'Brazil': (-15.780, -47.929), 'Argentina': (-34.604, -58.382),
    'Canada': (45.422, -75.697), 'Mexico': (19.433, -99.133), 'Colombia': (4.610, -74.082),
    'Peru': (-12.046, -77.043), 'UK': (51.507, -0.128), 'Germany': (52.520, 13.405),
    'Ireland': (53.333, -6.249), 'Netherlands': (52.368, 4.904), 'Italy': (41.903, 12.496),
    'Portugal': (38.717, -9.133), 'Austria': (48.208, 16.374), 'France': (48.857, 2.352),
    'Spain': (40.417, -3.703), 'Belgium': (50.850, 4.352), 'Switzerland': (46.948, 7.447),
    'Denmark': (55.676, 12.568), 'Sweden': (59.329, 18.069), 'Norway': (59.914, 10.752),
    'Finland': (60.170, 24.938), 'Poland': (52.230, 21.012), 'Czech': (50.076, 14.438),
    'India': (28.614, 77.209), 'Australia': (-33.869, 151.209), 'New_Zealand': (-40.9, 174.886),
    'South_Korea': (37.567, 126.978), 'Japan': (35.676, 139.650), 'China': (39.904, 116.407),
    'Vietnam': (21.028, 105.834), 'Thailand': (13.756, 100.502), 'Singapore': (1.352, 103.820),
    'Taiwan': (25.033, 121.565), 'Israel': (31.768, 35.214), 'Oman': (23.588, 58.383),
    'UAE': (24.454, 54.377), 'Qatar': (25.285, 51.531), 'Saudi_Arabia': (24.688, 46.722),
}

# Nigerian state tokens (any of these in a name -> Nigeria)
NIGERIA_STATES = {
    'Rivers','Bayelsa','Delta','Edo','AkwaIbom','Akwa-Ibom','AKWA_IBOM','CrossRiver',
    'Cross-River','CROSS_RIVER','Imo','Abia','Anambra','Enugu','Ebonyi','EBONYI','Ondo','ONDO',
    'Lagos','LAGOS','Ogun','OGUN','Oyo','OYO','Osun','Kwara','Benue','BENUE','Nasarawa',
    'NASARAWA','Plateau','PLATEAU','FCT','Fct','Niger','NIGER','Kogi','Kaduna','KADUNA',
    'Kaduna-south','Kano','Kebbi','KEBBI','Sokoto','SOKOTO','Zamfara','Borno','Gombe','GOMBE',
    'Taraba','Yobe','Adamawa','BAYELSA','Isiala','Isiala-North','Onu-east','Yenegoa','Bwari',
    'Lokoja','Mongono','Biyu','Abak','NRL','CPHL','NPHRL',
}
_TOKENS = {'TRM':'Cameroon','UKHSA':'UK','PSB':'UK','UL-30':'UK','RKI':'Germany',
           'ICMR':'India','MCL':'India','NVRL':'Ireland','INMI':'Italy','NPHRL':'Ghana'}
_FRAGS = [
    ('South_Korea','South_Korea'),('South Korea','South_Korea'),('Saudi_Arabia','Saudi_Arabia'),
    ('Sierra','Sierra_Leone'),('Cameroon','Cameroon'),('Cameroun','Cameroon'),
    ('Cote_dIvoire','Cote_dIvoire'),('Ivoire','Cote_dIvoire'),('IPCI','Cote_dIvoire'),
    ('Netherlands','Netherlands'),('Switzerland','Switzerland'),('Singapore','Singapore'),
    ('Argentina','Argentina'),('Australia','Australia'),('Portugal','Portugal'),
    ('Thailand','Thailand'),('Vietnam','Vietnam'),('Belgium','Belgium'),('Denmark','Denmark'),
    ('Finland','Finland'),('Germany','Germany'),('Ireland','Ireland'),('Austria','Austria'),
    ('Brazil','Brazil'),('Canada','Canada'),('France','France'),('Guinea','Guinea'),
    ('Israel','Israel'),('Mexico','Mexico'),('Norway','Norway'),('Poland','Poland'),
    ('Sweden','Sweden'),('Taiwan','Taiwan'),('Ghana','Ghana'),('Italy','Italy'),
    ('Japan','Japan'),('China','China'),('India','India'),('Spain','Spain'),
    ('Togo','Togo'),('Zambia','Zambia'),('Oman','Oman'),('Peru','Peru'),
    ('Qatar','Qatar'),('UAE','UAE'),('USA','USA'),('UK','UK'),
]

def _resolve_country(name: str) -> Optional[str]:
    s = name
    for p in ('hMpxV', 'hmpxv', 'MPXV', 'mpxv'):
        if s.startswith(p): s = s[len(p):]; break
    if s.startswith('|'): s = s[1:]
    for part in name.split('|'):
        p = part.strip()
        if p in ALL_COUNTRIES: return p
        if p in _ALIASES: return _ALIASES[p]
    for part in name.split('|'):
        if part.strip() in NIGERIA_STATES: return 'Nigeria'
    if 'Nigeria' in name: return 'Nigeria'
    for tok, ctry in _TOKENS.items():
        if tok in name: return ctry
    for frag, ck in _FRAGS:
        if frag in name: return ck
    return None

def colour_of(name, highlight):
    c = _resolve_country(name)
    if c == 'Nigeria':
        if NIGERIA_GREY:
            return NIGERIA_NEW_GREY if name in highlight else NIGERIA_OLD_GREY
        return AFRICA_COLOURS['Nigeria']
    if c in AFRICA_COLOURS:     return AFRICA_COLOURS[c]
    if c in EUROPE_COUNTRIES:   return CONTINENT_COLOURS['Europe']
    if c in ASIA_COUNTRIES:     return CONTINENT_COLOURS['Asia']
    if c in AMERICAS_COUNTRIES: return CONTINENT_COLOURS['Americas']
    return CONTINENT_COLOURS['Other']

# ===================== tree helpers =====================
def safe_draw_tree(tree):
    counter = [0]
    def _assign(node):
        if node.is_leaflike():
            counter[0] += 1; node.y = float(counter[0])
        else:
            for child in node.children:
                _assign(child)
            node.y = float(np.mean([c.y for c in node.children]))
    _assign(tree.root)
    all_y = [k.y for k in tree.Objects]
    tree.ySpan = max(all_y) - min(all_y) if len(all_y) > 1 else 1.0

def prune_leaves_from_tree(tree, remove_names):
    remove_set = set(remove_names)
    def _keep(node):
        if node.branchType == 'leaf':
            return node.name not in remove_set
        node.children = [c for c in node.children if _keep(c)]
        return len(node.children) > 0
    _keep(tree.root)
    objs = []
    def _collect(node):
        if node is not tree.root:
            objs.append(node)
        if node.branchType != 'leaf':
            for c in node.children:
                _collect(c)
    _collect(tree.root)
    tree.Objects = objs
    tree.treeHeight = max(k.height for k in objs) if objs else 0

# ===================== polar transform (full-world map, all countries) =====================

def _normalise_lineage_label(x):
    """Clean Nextclade lineage labels. Treat unassigned as the zoonotic group."""
    x = '' if pd.isna(x) else str(x).strip()
    if x == '' or x.lower() in {'nan', 'none', 'unassigned'}:
        return 'Zoonotic'
    return x


def _load_nextclade_lineages(nextclade_tsv):
    """Return {seqName: lineage} from Nextclade output."""
    if not nextclade_tsv or not os.path.exists(nextclade_tsv):
        print(f'[lineage labels] Nextclade TSV not found: {nextclade_tsv}')
        return {}
    df = pd.read_csv(nextclade_tsv, sep='\t', dtype=str)
    if 'seqName' not in df.columns or 'lineage' not in df.columns:
        print('[lineage labels] Nextclade TSV must contain seqName and lineage columns')
        return {}

    clean_lineage = df['lineage'].map(_normalise_lineage_label)
    lineage_map = dict(zip(df['seqName'].astype(str).str.strip(), clean_lineage))
    counts = clean_lineage.value_counts().to_dict()
    print(f'[lineage labels] loaded {len(lineage_map):,} lineage calls from {nextclade_tsv}')
    print('[lineage labels] lineages:', counts)
    return lineage_map


def _lineage_of_tip(name, lineage_map):
    """Exact lookup first; then a tolerant lookup for names with small formatting changes."""
    if not lineage_map:
        return ''
    if name in lineage_map:
        return lineage_map[name]
    n = str(name).strip()
    if n in lineage_map:
        return lineage_map[n]
    # Tolerant fallback: compare the accession/name part before the first pipe.
    prefix = n.split('|')[0]
    for k, v in lineage_map.items():
        if str(k).split('|')[0] == prefix:
            return v
    return ''


def _upright_tangent_rotation(angle):
    """
    Rotation for a complete label placed tangent to a circular arc.

    This keeps the lineage label as one readable word/code instead of drawing
    each character separately. The label is flipped where necessary so it is
    never upside down.
    """
    rot = -np.rad2deg(angle) + 90
    while rot < -90:
        rot += 180
    while rot > 90:
        rot -= 180
    return rot

def _lineage_label_alignment(angle):
    """Horizontal alignment for readable labels placed outside the arc."""
    x = np.sin(angle)
    if globals().get('LINEAGE_LABEL_MODE', 'horizontal') != 'horizontal':
        return 'center'
    if x > 0.15:
        return 'left'
    if x < -0.15:
        return 'right'
    return 'center'


def _outward_from_ring(ring_r, pad):
    """
    Move farther OUTSIDE from a circular ring in this Gytis-style polar layout.

    Important:
    The tree uses negative radii after inwardSpace is applied. Matplotlib draws a
    negative radius at the opposite angular direction, but the visible distance
    from the centre is abs(ring_r). Therefore:
      - if ring_r is positive, outside means +pad
      - if ring_r is negative, outside means -pad
    """
    return ring_r + (pad if ring_r >= 0 else -pad)


def _angle_for_readable_text(angle):
    """Return a tangent rotation and whether the letters should be reversed.

    Matplotlib cannot natively bend a whole word along a circular path, so we
    draw one character at a time. On the left/bottom side of the circle the
    natural tangent direction would make words upside down; reversing the
    character order keeps labels readable.
    """
    rot = -np.rad2deg(angle) + 90
    reverse = False
    if rot < -90 or rot > 90:
        rot += 180
        reverse = True
    while rot < -90:
        rot += 180
        reverse = not reverse
    while rot > 90:
        rot -= 180
        reverse = not reverse
    return rot, reverse


def _draw_curved_text_on_arc(ax, text, mid_angle, radius, colour, fontsize,
                             zorder=31, char_spacing=0.013, halo_width=3.0):
    """Draw text character-by-character so it follows the circular arc.

    Parameters
    ----------
    mid_angle : float
        Centre angle of the label, in radians, using the same convention as the
        tree: x = sin(angle) * radius, y = cos(angle) * radius.
    radius : float
        Signed radius. This matters for the Gytis-style inward tree layout,
        where the displayed outer ring can have a negative radius.
    char_spacing : float
        Approximate angular spacing per character at radius 1. Increase this if
        letters overlap; decrease it if labels become too long.
    """
    if not text:
        return

    # Use absolute radial size for spacing, but keep the signed radius for the
    # coordinates. This fixes labels when the tree uses negative radii.
    rr = max(abs(radius), 1e-6)
    chars = list(text)

    # Scale spacing gently with font size and radius. Wider labels are spread
    # over a longer angular span while remaining centred on the lineage arc.
    step = char_spacing * (fontsize / 25.0) / rr

    _, reverse = _angle_for_readable_text(mid_angle)
    if reverse:
        chars = chars[::-1]

    offsets = (np.arange(len(chars)) - (len(chars) - 1) / 2.0) * step

    for ch, off in zip(chars, offsets):
        aa = mid_angle + off
        x = np.sin(aa) * radius
        y = np.cos(aa) * radius
        rot, _ = _angle_for_readable_text(aa)
        ax.text(
            x, y, ch,
            ha='center', va='center',
            rotation=rot, rotation_mode='anchor',
            fontsize=fontsize, color=colour, zorder=zorder, clip_on=False,
            path_effects=[]
        )


def _ordered_lineages(lineage_counts):
    """Order important lineages first, then append all remaining detected lineages."""
    ordered = [lin for lin in LINEAGE_LABEL_ORDER if lin in lineage_counts]
    remaining = [lin for lin, _ in lineage_counts.most_common()
                 if lin not in set(ordered)]
    return ordered + remaining


def _lineage_colour(lin, i=0):
    """Colour known lineages manually; assign stable fallback colours to any extras."""
    if lin in LINEAGE_LABEL_COLOURS:
        return LINEAGE_LABEL_COLOURS[lin]
    cmap = plt.get_cmap('tab20')
    return mpl.colors.to_hex(cmap(i % cmap.N))


def _draw_lineage_list(ax, lineage_counts):
    """Draw all detected lineages as a clean, frame-free legend/list."""
    if not SHOW_LINEAGE_LIST or not lineage_counts:
        return

    ordered = _ordered_lineages(lineage_counts)
    handles = []
    for i, lin in enumerate(ordered):
        label = LINEAGE_LABEL_NAMES.get(lin, lin)
        if LINEAGE_LIST_SHOW_COUNTS:
            label = f'{label} ({lineage_counts[lin]})'
        handles.append(Line2D(
            [0], [0], marker='s', linestyle='None',
            markerfacecolor=_lineage_colour(lin, i), markeredgecolor='none',
            markersize=14, label=label
        ))

    leg = ax.legend(
        handles=handles,
        title=LINEAGE_LIST_TITLE,
        loc=LINEAGE_LIST_LOC,
        bbox_to_anchor=LINEAGE_LIST_BBOX,
        bbox_transform=ax.transAxes,
        ncol=LINEAGE_LIST_NCOL,
        frameon=LINEAGE_LIST_FRAMEON,
        fontsize=LINEAGE_LIST_FONT_SIZE,
        title_fontsize=LINEAGE_LIST_FONT_SIZE,
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    # Ensure no frame/background is visible even if matplotlib defaults change.
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_alpha(0)
    ax.add_artist(leg)


def _draw_lineage_arcs(ax, lineage_records, normaliseHeight, tree_height, inwardSpace, outer_scale_r=None):
    """
    Draw lineage annotations outside the final mutation-scale ring.

    The key design choice is that the arc baseline is not the tree-tip radius
    and not the central-map radius. It is the radius of the OUTERMOST mutation
    scale circle, passed in as `outer_scale_r`. Each lineage arc is drawn just
    beyond that last ring, and its label is placed still farther outside.
    """
    if not lineage_records:
        return

    by_lineage = {}
    for rec in lineage_records:
        lin = _normalise_lineage_label(rec.get('lineage', ''))
        if not lin:
            continue
        rec = dict(rec)
        rec['lineage'] = lin
        by_lineage.setdefault(lin, []).append(rec)

    lineage_counts = Counter({lin: len(items) for lin, items in by_lineage.items()})

    # Draw the clean all-lineage list first. This is independent of arc drawing.
    _draw_lineage_list(ax, lineage_counts)

    # Requested default: no outer arcs/ribbons.
    if not SHOW_LINEAGE_ARCS:
        return

    max_gap = np.deg2rad(LINEAGE_ARC_GAP_DEGREES)

    # Use the outermost mutation-scale ring as the anchor for lineage arcs.
    # If this function is called without outer_scale_r, fall back to just
    # outside the full tree height, but the normal call passes the true scale
    # radius from the scale-ring calculation.
    scale_r = outer_scale_r if outer_scale_r is not None else normaliseHeight(tree_height + inwardSpace)

    # IMPORTANT: in this Gytis-style layout the visible outer mutation ring can
    # have a NEGATIVE radius. Adding +pad to a negative radius moves the arc
    # inward, which is why the previous labels appeared inside the tree/map.
    # This sign keeps every lineage arc outside the last mutation ring.
    outside_sign = 1.0 if scale_r >= 0 else -1.0

    ordered_lineages = _ordered_lineages(lineage_counts)

    label_specs = []          # filled in the loop, placed after it by _place_lineage_labels

    # Draw in defined/detected order so important lineages get stable tracks.
    for track_i, lin in enumerate(ordered_lineages):
        items = by_lineage.get(lin, [])
        if len(items) < LINEAGE_ARC_MIN_TIPS:
            continue

        segments = _split_angular_clusters(items, max_gap)
        segments = [seg for seg in segments if len(seg) >= LINEAGE_ARC_MIN_TIPS]
        if not segments:
            continue
        # Which segment carries the label. Tip count first, then proximity to
        # the lineage's own family — the sublineages whose names nest with this
        # one (B.1 with B.1.2 and B.1.23; A.1.1 with A.1). Without the second
        # term, a lineage split between two equally small segments is labelled
        # on whichever one the circle happens to reach first, which put B.1's
        # label out among the zoonotic tips with a leader line running back
        # across the Nigerian lineages, rather than next to A.1.1 where the rest
        # of the B family sits.
        _fam = [l2 for l2 in by_lineage
                if l2 != lin and (l2.startswith(lin + '.') or lin.startswith(l2 + '.'))]
        _fam_angles = [x['angle'] for l2 in _fam for x in by_lineage[l2]]
        if _fam_angles and len(segments) > 1:
            _fc = _circular_mean(_fam_angles)

            def _seg_key(seg):
                m = _circular_mean([x['angle'] for x in seg])
                return (-len(seg), abs((m - _fc + np.pi) % (2 * np.pi) - np.pi))

            segments = sorted(segments, key=_seg_key)[:LINEAGE_MAX_SEGMENTS_PER_LINEAGE]
        else:
            segments = sorted(segments, key=len, reverse=True)[:LINEAGE_MAX_SEGMENTS_PER_LINEAGE]

        col = _lineage_colour(lin, track_i)

        # Put arcs OUTSIDE the last mutation-scale circle. The sign-aware
        # radius is essential when the mutation rings use negative radii.
        base_r = scale_r + outside_sign * (
            LINEAGE_ARC_RADIUS_PAD + (track_i % 8) * LINEAGE_ARC_TRACK_SPACING
        )

        for seg_i, seg in enumerate(segments):
            angles = sorted([x['angle'] for x in seg])
            a0, a1 = min(angles), max(angles)
            if abs(a1 - a0) < np.deg2rad(2.0):
                mid = 0.5 * (a0 + a1)
                a0, a1 = mid - np.deg2rad(1.0), mid + np.deg2rad(1.0)
            pad = np.deg2rad(0.75)
            arc_angles = np.linspace(a0 - pad, a1 + pad, 160)
            r = _outward_from_ring(base_r, seg_i * LINEAGE_ARC_TRACK_SPACING * 0.45)

            ax.plot(np.sin(arc_angles) * r, np.cos(arc_angles) * r,
                    color=col, lw=LINEAGE_ARC_LINEWIDTH, alpha=LINEAGE_ARC_ALPHA,
                    solid_capstyle='round', zorder=LINEAGE_ARC_ZORDER, clip_on=False)

            # Small terminal ticks make the arc read like a bracket.
            for aa in (a0 - pad, a1 + pad):
                ax.plot([np.sin(aa) * (r - 0.014), np.sin(aa) * (r + 0.014)],
                        [np.cos(aa) * (r - 0.014), np.cos(aa) * (r + 0.014)],
                        color=col, lw=LINEAGE_ARC_LINEWIDTH * 0.55,
                        alpha=LINEAGE_ARC_ALPHA, solid_capstyle='round', zorder=LINEAGE_ARC_ZORDER, clip_on=False)

            mid = 0.5 * (a0 + a1)
            arc_span_deg = np.rad2deg(abs((a1 + pad) - (a0 - pad)))
            if arc_span_deg < globals().get('LINEAGE_LABEL_MIN_SPAN_DEGREES', 2.0):
                continue

            # Use short lineage codes on the arc. Long descriptive names are
            # unreadable on circular figures at 25 pt and should stay in the
            # manuscript or figure legend.
            label = lin if globals().get('LINEAGE_ARC_LABEL_USE_SHORT_CODE', True) else LINEAGE_LABEL_NAMES.get(lin, lin)

            label_r = _outward_from_ring(r, LINEAGE_LABEL_RADIAL_PAD)

            # Small radial staggering avoids neighbouring labels sitting directly
            # on top of each other while keeping the map/tree size unchanged.
            if globals().get('LINEAGE_LABEL_STAGGER', True):
                label_r = _outward_from_ring(label_r, (track_i % 3) * 0.008)

            dx, dy = LINEAGE_LABEL_NUDGE.get(lin, (0.0, 0.0))
            lx = np.sin(mid) * label_r + dx
            ly = np.cos(mid) * label_r + dy
            rot = _upright_tangent_rotation(mid)

            # Label only the largest visible segment per lineage by default.
            if LINEAGE_LABEL_EVERY_SEGMENT or seg_i == 0:
                label_specs.append(dict(text=label, angle=mid, radius=label_r,
                                        arc_radius=r, colour=col, nudge=(dx, dy),
                                        n_tips=len(items)))

    _place_lineage_labels(ax, label_specs, outside_sign)


def _lineage_sort_key(name):
    """Nomenclature order: A.1 before A.1.1 before B.1 before B.1.2, F.4 last.

    Used to order the rungs of a label ladder. The ladder is a list of labels,
    each leadered back to its own arc, so ordering it by the lineage hierarchy
    rather than by angle puts the descent a reader follows — Nigerian A.x, then
    A.1.1, then B.1 and its sublineages — in the order they are named.
    """
    parts = str(name).split('.')
    head = parts[0].upper()
    nums = []
    for p in parts[1:]:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    return (head, nums)


def _place_lineage_labels(ax, specs, outside_sign=1.0):
    """Draw collected lineage labels, separating any that would overlap.

    Each label's true angular width is measured from the renderer at its own
    radius, then labels are walked in angular order: whenever the gap to the
    previously placed label is smaller than both labels need, the label is
    pushed along the arc and alternated onto a second radial track, and a
    leader line is drawn from its arc back to the moved text.
    """
    if not specs:
        return []
    halo = []          # no stroke: path effects would outline the glyphs

    # Freeze the view BEFORE measuring. Text is sized in points, so if the axes
    # rescale afterwards the text grows relative to the data and resolved
    # overlaps come back. A small symmetric margin reserves room for the ladder.
    ax.relim()
    ax.autoscale_view()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    mx, my = LINEAGE_LABEL_CANVAS_MARGIN * (x1 - x0), LINEAGE_LABEL_CANVAS_MARGIN * (y1 - y0)
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(y0 - my, y1 + my)
    ax.set_autoscale_on(False)

    def draw(sp, angle, radius):
        return ax.text(np.sin(angle) * radius + sp['nudge'][0],
                       np.cos(angle) * radius + sp['nudge'][1], sp['text'],
                       ha='center', va='center',
                       rotation=_upright_tangent_rotation(angle),
                       rotation_mode='anchor', fontsize=LINEAGE_TEXT_SIZE,
                       fontweight='regular', color=sp['colour'],
                       zorder=LINEAGE_LABEL_ZORDER, clip_on=False,
                       path_effects=halo)

    arts = [draw(sp, sp['angle'], sp['radius']) for sp in specs]
    if not LINEAGE_LABEL_DECOLLIDE:
        return arts

    # measure: angular width each label needs at its own radius
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for sp, art in zip(specs, arts):
        bb = art.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        # rotated tangentially, so the long side lies along the arc
        width = max(abs(x1 - x0), abs(y1 - y0))
        sp['need'] = LINEAGE_LABEL_GAP_PAD * width / max(abs(sp['radius']), 1e-6)
    # one text line's height in data units, from the font size and the transform
    (_a, _b), (_c, _d) = inv.transform([[0.0, 0.0], [0.0, 100.0]])
    data_per_px = abs(_d - _b) / 100.0
    line_h = LINEAGE_TEXT_SIZE * fig.dpi / 72.0 * data_per_px

    # cluster labels whose angular gaps are too small
    order = sorted(range(len(specs)), key=lambda i: specs[i]['angle'])
    clusters, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        need = 0.5 * (specs[a]['need'] + specs[b]['need'])
        if specs[b]['angle'] - specs[a]['angle'] < need:
            cur.append(b)
        else:
            clusters.append(cur); cur = [b]
    clusters.append(cur)

    for cl in clusters:
        if len(cl) == 1:
            i = cl[0]
            specs[i].update(final_angle=specs[i]['angle'], final_track=0, horizontal=False)
            continue
        centre = float(np.mean([specs[i]['angle'] for i in cl]))
        if len(cl) >= LINEAGE_LABEL_LADDER_MIN:
            # Dense cluster: a radial ladder of HORIZONTAL labels, ordered by
            # angle, each stepped further out and leadered back to its arc.
            # Horizontal text stacks cleanly where tangential text cannot, and
            # its bounding box is exact so the overlap check is meaningful.
            _rungs = sorted(cl, key=lambda j: _lineage_sort_key(specs[j]['text']))
            print('[lineage labels] ladder rungs:',
                  ' -> '.join(specs[j]['text'] for j in _rungs))
            # every rung steps out from ONE shared radius - the outermost arc in
            # the cluster - otherwise each label keeps its own arc's radius and
            # the rung order does not survive as the reading order
            _base = max((specs[i]['radius'] for i in cl), key=abs)
            for k, i in enumerate(_rungs):
                specs[i].update(final_angle=centre, final_track=k + 1,
                                horizontal=True, rung=k + 1, ladder_radius=_base)
        else:
            # Two or three labels: fan them symmetrically, alternating tracks.
            n_tracks = 2
            for k, i in enumerate(cl):
                specs[i]['final_track'] = k % n_tracks
                specs[i]['horizontal'] = False
            for t in range(n_tracks):
                mem = [i for i in cl if specs[i]['final_track'] == t]
                if not mem:
                    continue
                step = max(specs[i]['need'] for i in mem)
                offs = (np.arange(len(mem)) - (len(mem) - 1) / 2.0) * step
                for i, off in zip(mem, offs):
                    specs[i]['final_angle'] = centre + off

    def _xy(sp, extra=0.0):
        t = sp['final_track']
        if sp.get('horizontal'):
            step, start = LINEAGE_LABEL_LADDER_PAD * line_h, LINEAGE_LABEL_LADDER_START
        else:
            step, start = LINEAGE_LABEL_TRACK_STEP, 0.0
        r = _outward_from_ring(sp.get('ladder_radius', sp['radius']),
                               start + t * step + extra)
        a = sp['final_angle']
        return np.sin(a) * r + sp['nudge'][0], np.cos(a) * r + sp['nudge'][1]

    leaders, moved = [], 0
    for sp, art in zip(specs, arts):
        sp['extra'] = 0.0
        unmoved = (abs(sp['final_angle'] - sp['angle']) <= np.deg2rad(0.15)
                   and not sp['final_track'] and not sp.get('horizontal'))
        if unmoved:
            leaders.append(None)
            continue
        moved += 1
        lx, ly = _xy(sp)
        art.set_position((lx, ly))
        art.set_rotation(0 if sp.get('horizontal')
                         else _upright_tangent_rotation(sp['final_angle']))
        if sp.get('horizontal'):
            art.set_ha('center')
            art.set_fontsize(LINEAGE_TEXT_SIZE * LINEAGE_LABEL_LADDER_FONT)
        ar = _outward_from_ring(sp['arc_radius'], 0.012)
        ln, = ax.plot([np.sin(sp['angle']) * ar, lx * 0.988],
                      [np.cos(sp['angle']) * ar, ly * 0.988],
                      color=sp['colour'], lw=LINEAGE_LABEL_LEADER_LW,
                      alpha=LINEAGE_LABEL_LEADER_ALPHA, solid_capstyle='round',
                      zorder=LINEAGE_LABEL_ZORDER - 1, clip_on=False)
        leaders.append(ln)

    # Residual pass: laddering can push a label into one from a neighbouring
    # cluster. Step the SMALLER lineage further out until none are left. The
    # view is then grown to contain every label; because that rescale makes
    # point-sized text relatively larger, the two are alternated to convergence.
    def _resolve():
        for n in range(14):
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            boxes = [a.get_window_extent(rend) for a in arts]
            clash = [(i, j) for i in range(len(arts)) for j in range(i + 1, len(arts))
                     if boxes[i].overlaps(boxes[j])]
            if not clash:
                return n
            for i, j in clash:
                k = i if specs[i]['n_tips'] <= specs[j]['n_tips'] else j
                sp, art = specs[k], arts[k]
                sp['extra'] = sp.get('extra', 0.0) + 0.62 * line_h
                lx, ly = _xy(sp, sp['extra'])
                art.set_position((lx, ly))
                ar = _outward_from_ring(sp['arc_radius'], 0.012)
                if leaders[k] is None:
                    ln, = ax.plot([np.sin(sp['angle']) * ar, lx * 0.988],
                                  [np.cos(sp['angle']) * ar, ly * 0.988],
                                  color=sp['colour'], lw=LINEAGE_LABEL_LEADER_LW,
                                  alpha=LINEAGE_LABEL_LEADER_ALPHA,
                                  solid_capstyle='round',
                                  zorder=LINEAGE_LABEL_ZORDER - 1, clip_on=False)
                    leaders[k] = ln
                else:
                    leaders[k].set_data([np.sin(sp['angle']) * ar, lx * 0.988],
                                        [np.cos(sp['angle']) * ar, ly * 0.988])
        return 14

    def _fit_view():
        """Grow the limits until every label sits inside; True if changed."""
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        xs, ys = [], []
        for a in arts:
            bb = a.get_window_extent(rend)
            (ax0, ay0), (ax1, ay1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            xs += [ax0, ax1]; ys += [ay0, ay1]
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        pad = 0.35 * line_h
        nx0, nx1 = min(x0, min(xs) - pad), max(x1, max(xs) + pad)
        ny0, ny1 = min(y0, min(ys) - pad), max(y1, max(ys) + pad)
        if (nx0, nx1, ny0, ny1) == (x0, x1, y0, y1):
            return False
        ax.set_xlim(nx0, nx1); ax.set_ylim(ny0, ny1)
        return True

    iters = []
    for attempt in range(4):
        iters.append(_resolve())
        if not _fit_view():
            break
    it = sum(iters)
    print(f'[lineage labels] residual overlap pass: {it} iteration(s)')
    print(f'[lineage labels] {len(specs)} labels, {moved} repositioned to clear overlaps')
    return arts


def _tip_marker_size(ctry, is_hi):
    """Keep new Nigerian sequences biggest; make all other tips larger than before."""
    if is_hi and ctry == 'Nigeria':
        return NEW_NIGERIAN_TIP_SIZE
    if is_hi:
        # Highlighted but non-Nigerian, if any; still smaller than new Nigerian tips.
        return min(NEW_NIGERIAN_TIP_SIZE * 0.82, 180)
    if ctry in AFRICA_COLOURS:
        return AFRICA_TIP_SIZE
    return TIP_SIZE


def _split_angular_clusters(items, max_gap_radians):
    """
    Split items for the same map location into visually local clusters.
    Each item must have an 'angle' key. This avoids connecting far-apart tips
    into one misleading convergence point.
    """
    if not items:
        return []
    if len(items) == 1:
        return [items]

    ordered = sorted(items, key=lambda d: d['angle'])
    clusters = [[ordered[0]]]
    for item in ordered[1:]:
        if abs(item['angle'] - clusters[-1][-1]['angle']) <= max_gap_radians:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def _circular_mean(angles):
    """Mean direction for angles in radians."""
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _draw_clustered_connectors(ax, ax2, proj, connectors, done_coords, normaliseHeight,
                               tree_height, inwardSpace):
    """
    Draw tip connectors in two stages:
      1. nearby tips from the same country/location converge to one external point;
      2. one clean line links that convergence point to the country marker on the map.
    """
    if not connectors:
        return

    # Group by resolved country/map coordinate. This means a Cameroon cluster, UK cluster,
    # Nigeria cluster, etc. each gets its own convergence point and one line to the map.
    by_country = {}
    for rec in connectors:
        by_country.setdefault(rec['ctry'], []).append(rec)

    max_gap = np.deg2rad(CLUSTER_ANGLE_GAP_DEGREES)
    rim_r = normaliseHeight(tree_height * 1.025 + inwardSpace)
    anchor_r = rim_r + CONNECTOR_RADIAL_PAD

    for ctry, items in by_country.items():
        ck = ctry if ctry in COORDINATES else (ctry.replace(' ', '_') if ctry else None)
        if ck not in COORDINATES:
            continue

        lat, lon = COORDINATES[ck]
        try:
            lon_p, lat_p = proj.transform_point(lon, lat, ccrs.PlateCarree())
        except Exception:
            continue
        if not (np.isfinite(lon_p) and np.isfinite(lat_p)):
            continue

        # Map marker once per coordinate.
        if (lon, lat) not in done_coords:
            if ctry == 'Nigeria':
                halo_s, mark_s = NIGERIA_MAP_HALO_SIZE, NIGERIA_MAP_MARKER_SIZE
            else:
                halo_s, mark_s = MAP_MARKER_HALO_SIZE, MAP_MARKER_SIZE
            ax2.scatter(lon, lat, s=halo_s, facecolor='k', edgecolor='none', zorder=9999,
                        transform=ccrs.PlateCarree(), clip_on=False)
            ax2.scatter(lon, lat, s=mark_s, facecolor=items[0]['colour'], edgecolor='none', zorder=10000,
                        transform=ccrs.PlateCarree(), clip_on=False)
            done_coords.append((lon, lat))

        # Only connect the locations requested by the switches.
        eligible = (ctry in AFRICA_COLOURS) if LINE_ONLY_AFRICA else True
        if ctry == 'Nigeria' and not CONNECT_NIGERIA_TO_MAP:
            eligible = False
        if not eligible:
            continue

        clusters = _split_angular_clusters(items, max_gap) if CLUSTER_CONNECTORS else [[x] for x in items]

        for cluster in clusters:
            angles = [x['angle'] for x in cluster]
            mean_angle = _circular_mean(angles)
            ax_anchor = (np.sin(mean_angle) * anchor_r, np.cos(mean_angle) * anchor_r)

            # Use the most common/first colour in the cluster. If a cluster contains new Nigeria,
            # the first records are already coloured deep grey by colour_of().
            col = cluster[0]['colour']

            # Thin local spokes: tips converge to the shared anchor point.
            # This is the key change requested.
            if len(cluster) > 1:
                spoke_segments = [((x['tip_x'], x['tip_y']), ax_anchor) for x in cluster]
                ax.add_collection(LineCollection(
                    spoke_segments,
                    lw=CONNECTOR_LINEWIDTH,
                    color=col,
                    alpha=CONNECTOR_ALPHA,
                    capstyle='round',
                    zorder=5
                ))
                ax.scatter(ax_anchor[0], ax_anchor[1], s=CONNECTOR_CLUSTER_POINT_SIZE,
                           facecolor=col, edgecolor='k', linewidth=0.4, zorder=6)
            else:
                # Single tip: keep a normal short radial segment to its external anchor.
                x = cluster[0]
                ax.plot([x['tip_x'], ax_anchor[0]], [x['tip_y'], ax_anchor[1]],
                        color=col, lw=CONNECTOR_LINEWIDTH, alpha=CONNECTOR_ALPHA, zorder=5)

            # One line from the convergence point to the map location.
            ax2.add_patch(ConnectionPatch(
                xyA=ax_anchor, coordsA=ax.transData, axesA=ax,
                xyB=(lon_p, lat_p), coordsB=ax2.transData, axesB=ax2,
                color=col, lw=CONNECTOR_LINEWIDTH, alpha=CONNECTOR_ALPHA, zorder=3
            ))


def polar_transform_global(ax, ax2, tree, proj, highlight_names=set(), lineage_map=None,
                           circStart=0.0, circFrac=1.0, inwardSpace=0.0, precision=200):
    if inwardSpace < 0:
        inwardSpace -= tree.treeHeight
    allXs = [k.height for k in tree.Objects]; allXs.append(max(allXs) * 1.1)
    _lo, _hi = min(allXs), max(allXs)
    normaliseHeight = lambda v: (v - _lo) / (_hi - _lo)
    lsp = lambda s, e, n: [s + ((e - s) / (n - 1)) * i for i in range(n)] if n > 1 else [e]
    circ_s = circStart * np.pi * 2
    circ   = circFrac * np.pi * 2
    branches, cs, lws = [], [], []
    done_coords = []
    connector_records = []
    lineage_records = []

    # ---- scale: faint full-circle rings spanning the WHOLE tree (branches cross them) ----
    def _nice_step(hi, n=7):
        raw = max(hi, 1) / n
        mag = 10 ** np.floor(np.log10(raw))
        for mlt in (1, 2, 2.5, 5, 10):
            if raw <= mlt * mag:
                return mlt * mag
        return 10 * mag
    step = _nice_step(tree.treeHeight)
    ticks = np.arange(0, tree.treeHeight + step, step)
    y0 = circ_s - 0.1                       # scale bar sits in the wedge gap of the tree
    x0n = normaliseHeight(0 + inwardSpace)
    x1n = normaliseHeight(ticks[-1] + inwardSpace)
    ax.plot([np.sin(y0)*x0n, np.sin(y0)*x1n], [np.cos(y0)*x0n, np.cos(y0)*x1n],
            color='k', lw=SCALE_BAR_LINEWIDTH, zorder=4)
    th = np.linspace(-np.pi, np.pi, 200)
    for j, t in enumerate(ticks):
        xn = normaliseHeight(t + inwardSpace)
        ax.plot(np.sin(th)*xn, np.cos(th)*xn, color='lightgrey', ls='--',
                alpha=0.55, lw=RING_LINEWIDTH, zorder=0)          # full ring crossing all branches
        w = 0.012
        yt = np.linspace(y0, y0 + w, 10)
        ax.plot(np.sin(yt)*xn, np.cos(yt)*xn, color='k', lw=SCALE_BAR_LINEWIDTH, zorder=4)
        ax.text(np.sin(yt[-1]+0.02)*xn, np.cos(yt[-1]+0.02)*xn, f'{int(t)}',
                ha='center', va='center', rotation=-np.rad2deg(y0),
                fontsize=SCALE_TEXT_SIZE, zorder=4)
    # the axis title sits well clear of the tick numbers, on the other side of
    # the scale bar, so the two do not cross
    _r_title = x0n + 0.105
    ax.text(np.sin(y0-0.055)*_r_title, np.cos(y0-0.055)*_r_title,
            'substitutions', ha='left', va='center', rotation=-np.rad2deg(y0)-90,
            fontsize=SCALE_TEXT_SIZE, zorder=4)

    for k in tree.Objects:
        xk  = normaliseHeight(k.height + inwardSpace)
        xkp = normaliseHeight(k.parent.height + inwardSpace) if k.parent.parent else xk
        yk  = circ_s + circ * k.y / tree.ySpan
        X, Y = np.sin(yk), np.cos(yk)

        colour = 'k'
        if k.branchType == 'leaf':
            colour = colour_of(k.name, highlight_names)
        cs.append(colour); lws.append(BRANCH_LINEWIDTH)
        branches.append(((X*xkp, Y*xkp), (X*xk, Y*xk)))

        if k.is_node():
            yl = circ_s + circ * k.children[0].y  / tree.ySpan
            yr = circ_s + circ * k.children[-1].y / tree.ySpan
            ybar = lsp(yl, yr, precision)
            xs = [np.sin(a)*xk for a in ybar]; ys = [np.cos(a)*xk for a in ybar]
            branches += list(zip(zip(xs, ys), zip(xs[1:], ys[1:])))
            lws += [NODE_ARC_LINEWIDTH]*len(xs[1:]); cs += [colour]*len(xs[1:])
        else:
            ctry = _resolve_country(k.name)
            is_hi = k.name in highlight_names
            tip_s = _tip_marker_size(ctry, is_hi)

            # Bigger tips, while new Nigerian sequences remain the largest.
            if is_hi or ctry == 'Nigeria' or ctry in AFRICA_COLOURS:
                ax.scatter(X*xk, Y*xk, s=tip_s * TIP_HALO_FACTOR, facecolor='k',
                           edgecolor='none', zorder=9999)
            ax.scatter(X*xk, Y*xk, s=tip_s, facecolor=colour,
                       edgecolor='none', zorder=10000)

            # Collect lineage placement info for outer lineage ribbons/labels.
            lin = _lineage_of_tip(k.name, lineage_map or {})
            if lin:
                lineage_records.append({
                    'name': k.name,
                    'lineage': lin,
                    'angle': yk,
                    'ctry': ctry,
                    'is_hi': is_hi,
                })

            # Collect connector candidates; draw them after all tips are known so nearby
            # tips can be merged into one convergence point before linking to the map.
            ck = ctry if ctry in COORDINATES else (ctry.replace(' ', '_') if ctry else None)
            if ck in COORDINATES:
                rim_r = normaliseHeight(tree.treeHeight * 1.025 + inwardSpace)
                connector_records.append({
                    'name': k.name,
                    'ctry': ctry,
                    'colour': colour,
                    'angle': yk,
                    'tip_x': X*xk,
                    'tip_y': Y*xk,
                    'rim_x': X*rim_r,
                    'rim_y': Y*rim_r,
                    'is_hi': is_hi,
                })

    # Draw tree branches first, then connectors above branches but below tip dots.
    ax.add_collection(LineCollection(branches, lw=[l * BRANCH_HALO_MULT for l in lws],
                                     color='w', capstyle='projecting', zorder=1))
    ax.add_collection(LineCollection(branches, lw=lws, color=cs,
                                     capstyle='projecting', zorder=2))

    _draw_clustered_connectors(ax, ax2, proj, connector_records, done_coords,
                               normaliseHeight, tree.treeHeight, inwardSpace)

    # Creative lineage labels: arcs follow the OUTERMOST mutation-scale ring.
    # `x1n` is the radius of the last mutation ring drawn above, so lineage
    # arcs/labels remain outside the tree and outside the central map.
    # The OUTERMOST mutation ring is the zero-mutation ring, x0n.
    # x1n is usually the inner/high-mutation ring in this inward Gytis layout.
    # Use x0n so lineage arcs sit outside the full mutation-scale circle.
    _draw_lineage_arcs(ax, lineage_records, normaliseHeight, tree.treeHeight, inwardSpace,
                       outer_scale_r=x0n)

# ===================== figure builder =====================
def make_circular_global_figure(treefile, outfile, highlight_csv=None, collapse_csv=None,
                                nextclade_tsv=None, sl_keep=20, aln_length=197209,
                                g1_names=None, fig=None, box=None, save=True,
                                drop_above_mutations=None,
                                figsize=(40, 30), map_width=0.40,
                                circStart=0.83, circFrac=0.97, gap_frac=0.85):
    highlight_names = set()
    if highlight_csv and os.path.exists(highlight_csv):
        highlight_names = set(pd.read_csv(highlight_csv)['seqName'].astype(str).str.strip())
        print(f'{len(highlight_names):,} highlighted (new) sequences')

    lineage_map = _load_nextclade_lineages(nextclade_tsv) if (SHOW_LINEAGE_ARCS or SHOW_LINEAGE_LIST) else {}

    with open(treefile) as _fh:
        _head = _fh.read(200).lstrip()
    my_tree = (bt.loadNexus(treefile, absoluteTime=False) if _head.upper().startswith('#NEXUS')
               else bt.loadNewick(treefile, absoluteTime=False))
    if aln_length > 1:
        for k in my_tree.Objects:
            k.length *= aln_length
    my_tree.traverse_tree()
    print(f'Tree height: {my_tree.treeHeight:.2f} mutations')

    # collapse: thin the Sierra Leone G.1 fan to sl_keep reps, but RETAIN every
    # other (unique-country) tip in that clade
    # Drop tips whose root-to-tip divergence is implausible for clade IIb: a
    # handful of sequences carry terminal branches two orders of magnitude
    # longer than the tree's median (1 mutation), which is an assembly/alignment
    # artefact, and leaving them in compresses the whole radial mutation scale
    # so the real branches become invisible. Reported, never silently dropped.
    if drop_above_mutations:
        safe_draw_tree(my_tree)
        far = {o.name: o.height for o in my_tree.Objects
               if o.branchType == 'leaf' and o.height > drop_above_mutations}
        if far:
            prune_leaves_from_tree(my_tree, set(far))
            my_tree.traverse_tree()
            print(f'divergence outliers pruned (>{drop_above_mutations:.0f} mutations '
                  f'from the root): {len(far)}')
            for n, h in sorted(far.items(), key=lambda kv: -kv[1]):
                print(f'    {h:8.1f}  {n}')
            print(f'Tree height after pruning: {my_tree.treeHeight:.1f} mutations')

    if g1_names is not None:
        all_g1 = set(g1_names)
    elif collapse_csv and os.path.exists(collapse_csv):
        all_g1 = set(pd.read_csv(collapse_csv)['seqName'].astype(str).str.strip())
    else:
        all_g1 = set()
    if True:
        if all_g1:
            safe_draw_tree(my_tree)
            g1 = sorted([o for o in my_tree.Objects
                         if o.branchType == 'leaf' and o.name in all_g1], key=lambda o: o.y)
            sl    = [l for l in g1 if _resolve_country(l.name) == 'Sierra_Leone']
            other = [l for l in g1 if _resolve_country(l.name) != 'Sierra_Leone']
            keep_sl = {sl[i].name for i in
                       np.round(np.linspace(0, len(sl)-1, min(sl_keep, len(sl)))).astype(int)} if sl else set()
            keep = keep_sl | {l.name for l in other}        # 20 SL + all unique-country tips
            remove = {l.name for l in g1 if l.name not in keep}
            prune_leaves_from_tree(my_tree, remove)
            uniq = sorted({_resolve_country(l.name) for l in other})
            print(f'G.1: {len(g1)} tips -> kept {len(keep_sl)} Sierra Leone + '
                  f'{len(other)} other ({len(uniq)} unique countries: {uniq}); removed {len(remove)}')
    safe_draw_tree(my_tree)

    _n_leaves = len([o for o in my_tree.Objects if o.branchType == 'leaf'])
    print(f'circular panel drawn with {_n_leaves} tips')
    inwardSpace = -(gap_frac * my_tree.treeHeight)   # tree-height-scaled (tips ring around map)

    # `box` = (left, bottom, width, height) in figure coordinates. When given,
    # the panel is drawn inside that rectangle of an existing figure instead of
    # filling its own, which is how it composes into the supplementary figure.
    own_fig = fig is None
    if own_fig:
        fig = plt.figure(figsize=figsize, facecolor='w')
    bx, by, bw, bh = box if box is not None else (0.0, 0.0, 1.0, 1.0)
    ax  = fig.add_axes([bx, by, bw, bh], zorder=1000, facecolor='none')

    # full-world map so America AND Asia are visible (Robinson, set_global)
    proj = ccrs.Robinson(central_longitude=MAP_CLON)
    mw   = map_width
    # map rectangle is relative to the panel box, so the central map keeps the
    # same fraction of the panel whatever size the panel is given
    # provisional rectangle; repositioned below from the tree's own transform so
    # the map is a true square centred on the tree, whatever shape the panel is
    ax2  = fig.add_axes([bx + bw * 0.3, by + bh * 0.3, bw * mw, bh * mw],
                        projection=proj, facecolor='none', zorder=0)
    # distinctive palette (warm land / pale-teal sea) — not Gytis's neutral grey
    water, land = '#D7E7EA', '#E2D7BE'
    ax2.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=water, edgecolor=water)
    ax2.add_feature(cfeature.LAND.with_scale('50m'),  facecolor=land,  edgecolor='w', linewidth=MAP_COAST_LINEWIDTH)
    ax2.add_feature(cfeature.LAKES.with_scale('50m'), facecolor=water)
    ax2.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='w', linewidth=MAP_BORDER_LINEWIDTH, zorder=1)
    ax2.set_global()
    for loc in ax2.spines:
        ax2.spines[loc].set_visible(False)

    polar_transform_global(ax, ax2, my_tree, proj, highlight_names=highlight_names,
                           lineage_map=lineage_map,
                           circStart=circStart, circFrac=circFrac,
                           inwardSpace=inwardSpace, precision=200)

    # ---- legends arranged in the corners around the tree (Gytis style) ----
    halo = []          # no stroke: path effects would outline the glyphs
    def corner_block(entries, x, y, dy=0.026, ha='left'):
        sw = 0.013
        for i, (col, lab) in enumerate(entries):
            yy = y - i * dy
            ax.add_patch(Rectangle((x, yy), sw, sw, transform=ax.transAxes,
                                   facecolor=col, edgecolor='none', clip_on=False, zorder=1e6))
            tx = x + sw * 1.5 if ha == 'left' else x - sw * 0.5
            ax.text(tx, yy + sw/2, lab, transform=ax.transAxes, ha=ha, va='center',
                    size=LEGEND_TEXT_SIZE, color='k', zorder=1e6, path_effects=halo)

    if NIGERIA_GREY:
        africa = [(NIGERIA_NEW_GREY, 'Nigeria \u2013 new'), (NIGERIA_OLD_GREY, 'Nigeria \u2013 older')]
    else:
        africa = [(AFRICA_COLOURS['Nigeria'], 'Nigeria')]
    africa += [(AFRICA_COLOURS[c], COUNTRY_DISPLAY.get(c, c.replace('_', ' ')))
               for c in ['Cameroon','Sierra_Leone','Togo','Guinea','Ghana','Cote_dIvoire','Zambia']]
    continents = [(CONTINENT_COLOURS[c], c) for c in ['Europe','Asia','Americas']] + \
                 [(CONTINENT_COLOURS['Other'], 'Other / Unknown')]

    corner_block(africa,     x=0.02, y=0.97, ha='left')    # West/Central Africa: top-left
    corner_block(continents, x=0.90, y=0.97, ha='right')   # rest of world: top-right

    for a in (ax, ax2):
        a.set_xticks([]); a.set_yticks([])
        for loc in a.spines:
            a.spines[loc].set_visible(False)
    ax.set_aspect('equal')

    # Keep the original tree/map scale. Do not expand x/y limits here;
    # otherwise the tree and central map shrink when lineage arcs are added.
    # The lineage arcs are drawn just outside the outer mutation ring within
    # the existing canvas.

    # Leave enough canvas for lineage arcs/labels outside the final mutation ring.
    if ax.get_autoscale_on():          # already frozen when lineage labels ran
        ax.relim()
        ax.autoscale_view()

    # The tree is drawn centred on the data origin with aspect='equal', so the
    # central map must be a display-space square about that origin covering the
    # same fraction of the tree as in the standalone figure.
    _half = 0.5 * mw * (ax.get_xlim()[1] - ax.get_xlim()[0])
    _p = ax.transData.transform([(-_half, -_half), (_half, _half)])
    (_fx0, _fy0), (_fx1, _fy1) = fig.transFigure.inverted().transform(_p)
    ax2.set_position([_fx0, _fy0, _fx1 - _fx0, _fy1 - _fy0])

    if not save:
        return fig, ax, ax2

    for ext, kw in [('pdf', {}), ('svg', {}),
                    ('png', {'dpi': 300, 'transparent': False})]:
        # Fixed canvas keeps the central map/tree the same size after adding outer labels.
        # Avoid bbox_inches='tight' because it rescales the final canvas around outer arcs.
        plt.savefig(f'{outfile}.{ext}', facecolor='w', **kw)
        print('saved', f'{outfile}.{ext}')


if __name__ == '__main__':
    make_circular_global_figure(
        treefile        = TREEFILE,
        outfile         = OUTFILE,
        highlight_csv   = HIGHLIGHT_CSV,
        collapse_csv    = COLLAPSE_CSV,
        nextclade_tsv   = NEXTCLADE_TSV,
        sl_keep         = SL_KEEP,
        aln_length      = MPOX_ALN_LENGTH,
        figsize         = (28, 28), map_width=MAP_WIDTH,
        circStart=0.83, circFrac=0.97, gap_frac=GAP_FRAC,
    )
