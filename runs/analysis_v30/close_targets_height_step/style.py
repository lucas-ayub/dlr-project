"""Plot style of the final presentation: Latin Modern Sans, red standard, purple joint, green reference."""
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ("lmsans10-regular.otf", "lmsans10-bold.otf", "lmsans10-oblique.otf"):
    try:
        fm.fontManager.addfont(subprocess.check_output(["kpsewhich", f]).decode().strip())
    except Exception:
        pass
INK, MUTED, GRID = "#1e293b", "#64748b", "#e2e8f0"
plt.rcParams.update({
    "font.family": "Latin Modern Sans", "mathtext.fontset": "cm", "font.size": 9.5,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 9, "xtick.labelsize": 8.8, "ytick.labelsize": 8.8,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#94a3b8", "xtick.color": "#475569",
    "ytick.color": "#475569", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.titlepad": 3.5, "axes.titlelocation": "left", "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
import os as _os
if _os.environ.get("FIG_STYLE") == "report":                 # serif fonts of the project report (report/scripts/make_figures.py)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm"})
COL = dict(ref="#2CA02C", std="#D62728", joint="#9467BD")
NAME = dict(std="standard reconstruction", joint="joint reconstruction", ref="reference")
TERRAIN, FILL, AMBER = "#785A3C", "#EBE4D6", "#D97706"
WL, R0 = 0.25, 766206.0
TH = np.arccos(720e3 / R0)
B300 = [0.0, 300.0, 119.8, 100.5]


def db(v, pk):
    return 20 * np.log10(np.abs(v) / pk + 1e-12)


def mismatch(b, dh, m, nrx=4):
    """Blind-height mismatch [rad] (as in the report, section on blind heights): rms over channels of the wrapped difference between the
    alias step 2 pi i m / N_rx and the height phase a_i dh; small = the place and its alias look alike in every channel."""
    a = 2 * np.pi * np.asarray(b) / (WL * R0 * np.tan(TH))
    e = np.angle(np.exp(1j * (2 * np.pi * np.arange(nrx) * m / nrx - a * dh)))
    return float(np.sqrt(np.mean(e ** 2)))


def save(fig, out, name):
    import os
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
