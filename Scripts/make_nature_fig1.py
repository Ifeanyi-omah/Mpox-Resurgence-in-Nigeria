"""Derive plot_figure1_nature.py from plot_figure1.py.

Content and layout are untouched; only the physical size, the type ladder,
line weights and the panel-title policy change, so the figure meets Nature
figure specification (183 mm double column, Arial, 5-7 pt type, 8 pt bold
lowercase panel letters, no claim titles, vector PDF with Type 42 fonts).
"""
import re

s = open("plot_figure1.py").read()

# ---- type ladder: 8 pt panel letters, 7 pt labels, 6 pt legends/in-panel, 5 pt ticks
def fs(v):
    v = float(v)
    return 8 if v >= 25 else (7 if v >= 19 else (6 if v >= 16 else 5))

s = re.sub(r"(fontsize|title_fontsize)=(\d+(?:\.\d+)?)",
           lambda m: "%s=%d" % (m.group(1), fs(m.group(2))), s)

# ---- line weights scale with the figure (20 in -> 7.2 in), 0.3 pt floor
s = re.sub(r"\b(linewidth|lw|markeredgewidth)=(\d+(?:\.\d+)?)",
           lambda m: "%s=%s" % (m.group(1), max(0.3, round(float(m.group(2)) * 0.42, 2))), s)
s = re.sub(r"\bmarkersize=(\d+(?:\.\d+)?)",
           lambda m: "markersize=%.1f" % max(2.0, float(m.group(1)) * 0.28), s)
# pads are in points: scale, but leave sub-point values (boxstyle pads) alone
s = re.sub(r"\b(pad|labelpad)=([2-9]\d*(?:\.\d+)?)",
           lambda m: "%s=%s" % (m.group(1), round(float(m.group(2)) * 0.4, 1)), s)

# ---- rcParams: explicit Nature ladder, thin spines, embedded fonts
s = s.replace("""        'font.size': 20, 'axes.labelsize': 20, 'axes.titlesize': 20,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 17,
        'axes.linewidth': 1.2, 'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
        'xtick.major.size': 6, 'ytick.major.size': 6,""",
"""        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 5, 'ytick.labelsize': 5, 'legend.fontsize': 6,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.2, 'ytick.major.size': 2.2,""")
s = s.replace("'font.sans-serif': ['Helvetica', 'Arial',",
              "'font.sans-serif': ['Arial', 'Helvetica',")

# ---- 183 mm double column; same aspect, so every axes rectangle stays valid
s = s.replace("figsize=(20, 25)", "figsize=(7.2, 9.0)")

# ---- Nature house style: claim titles move to the caption; wave identity stays
s = s.replace("    axE.set_title('Age distribution broadens across waves', fontsize=7, pad=3.2, loc='left')\n", "")
s = s.replace("axD.set_title('Wave 3 age-sex structure', fontsize=7, pad=3.2, loc='left')",
              "axD.set_title('Wave 3', fontsize=7, pad=3.2, loc='left')")
s = s.replace("axF.set_title('Testing does not explain the paediatric share', fontsize=7, pad=3.2, loc='left')",
              "axF.set_title('Wave 3', fontsize=7, pad=3.2, loc='left')")

# ---- value-on-every-bar is unreadable at 5 pt and redundant with the axis
s = s.replace("""        for xi, v in zip(xs + (i - 1) * bw, vals):
            axE.text(xi, v + 0.7, f'{v:.0f}', ha='center', va='bottom', fontsize=5, color=edge[w])
""", "")
s = s.replace("""        for xi, v in zip(xs + off, vals):
            axF.text(xi, v + 0.8, f'{v:.0f}', ha='center', va='bottom', fontsize=5, color='#333333')
""", "")
# tick labels stay on the bottom rung of the ladder
s = s.replace("axE.set_xticklabels(AGE_BANDS_PUB, fontsize=6)", "axE.set_xticklabels(AGE_BANDS_PUB, fontsize=5)")

# ---- exact 183 mm output width (a tight bbox would trim it) and 600 dpi raster
s = s.replace("""        fig.savefig(path, bbox_inches='tight', pad_inches=0.1, facecolor='white',
                    **({'dpi': args.dpi} if ext in ('png', 'tif') else {}))""",
"""        fig.savefig(path, facecolor='white',
                    **({'dpi': args.dpi} if ext in ('png', 'tif') else {}))""")
s = s.replace("default=400", "default=600")

# ---- colorbar ticks belong on the 5 pt rung
s = s.replace("cb.ax.tick_params(labelsize=14, length=4)",
              "cb.ax.tick_params(labelsize=5, length=1.6, width=0.5, pad=1.5)")

# ---- the zone legend was wider than the column once the tight bbox was dropped
s = s.replace("    axR.set_title('States and geopolitical zones', fontsize=7, pad=16.0, loc='center')\n", "")
s = s.replace("ncol=6, handlelength=1.2, columnspacing=1.4, handletextpad=0.5)",
              "ncol=3, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)")

open("plot_figure1_nature.py", "w").write(s)
print("wrote plot_figure1_nature.py", len(s))
