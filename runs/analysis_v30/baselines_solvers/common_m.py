"""Shared helpers of the v30 analyses M (time-varying baselines; regularisation methods).

Repository dependency: the sata1d package (sata1d_code/) and, for the SRTM lines, profiles.py of ../joint_height_dense (copied
next to this file as profiles.py when the folder is moved into the repository). The repository root comes from DLR_REPO
or is found by walking up from this file until a folder with sata1d_code/ appears.
"""
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo():
    r = os.environ.get("DLR_REPO")
    if r:
        return r
    d = HERE
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "sata1d_code")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repository root not found: set DLR_REPO")


REPO = _repo()
SC = os.path.join(REPO, "sata1d_code")
sys.path[:0] = [SC, os.path.join(SC, "joint_cases"), HERE]
import sata1d as S                                                     # noqa: E402
from sata1d.config import (ArrayGeometry, ExperimentConfig, Scene,      # noqa: E402
                           SystemParams, build_time_axis)
from sata1d.geometry import PlatformTracks                              # noqa: E402
from sata1d.reconstruction import GetCoeffNu                            # noqa: E402

CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "fig")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

B300 = [0.0, 300.0, 119.8, 100.5]

# deck colours
RED, PURPLE, GREEN, AMBER = "#D62728", "#7B3FA0", "#2CA02C", "#E07A3F"
C_C0, C_C1, C_C2 = "#E07A3F", "#156592", "#7C5CA8"
INK, MUTED, GRID = "#1e293b", "#64748b", "#e2e8f0"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.6, "axes.labelsize": 8.6, "axes.titlesize": 8.6, "legend.fontsize": 7.6,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": "#94a3b8", "xtick.color": "#475569", "ytick.color": "#475569", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.3, "axes.titlelocation": "left", "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "mathtext.fontset": "cm"})
    if os.environ.get("FIG_STYLE") == "report":               # serif fonts of the project report
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"]})
    return plt


def fs(w, h, w_report, h_report):
    """Figure size: the talk/analysis size, or the size for the text width of the report (FIG_STYLE=report)."""
    return (w_report, h_report) if os.environ.get("FIG_STYLE") == "report" else (w, h)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=170)


# ---------------------------------------------------------------- systems
def spaceborne_config(bxt=300.0):
    """The deck's default system, one Tx/Rx0 pair plus one receiver at cross-track baseline bxt (2 channels)."""
    cfg = S.make_config([(0.0, 0.0)], nrx=2, bxt_max=0.0, tx_rx_channel0=True)
    return replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.array([0.0, bxt])))


def airborne_config(wl=0.031, La=0.5, v=90.0, H=3000.0, look_deg=45.0, bxt=2.0, prf_factor=1.25, n_ap=3.0):
    """An airborne side-looking geometry (straight nominal track, flat earth: v_e = v).
    Defaults: X band (wavelength 3.1 cm), 0.5 m antenna, 90 m/s, 3000 m above the reference height, look angle 45 deg
    (slant range 4243 m), cross-track baseline 2 m; values typical of the DLR F-SAR airborne sensor (Do 228 platform,
    X/L band, flight altitude a few km; Reigber et al., Proc. IEEE 101(3), 2013). PRF = prf_factor * B_a."""
    r0 = H / np.cos(np.radians(look_deg))
    da_factor = La / 2.0 / wl
    sysp = SystemParams(wl=wl, ve=v, vs=v, da_factor=da_factor, La_factor=2.0)
    scene = Scene(rDelay=2.0 * r0 / sysp.c0, H=H, x0=0.0, h0=0.0, c0=sysp.c0)
    prf = prf_factor * sysp.abw
    t_ap = wl * r0 / La / v                     # one aperture [s]
    Na, Na_ch, ta = build_time_axis(prf, 2, n_ap * t_ap, divfac=64)
    return ExperimentConfig(system=sysp, scene=scene, array=ArrayGeometry(bat=np.zeros(2), bxt=np.array([0.0, bxt])),
                            prf=prf, PRF_op=prf / 2, Na=Na, Na_ch=Na_ch, ta=ta)


def tracks_with_eps(cfg, eps_t):
    """Tx on the nominal track; receiver 1 at cross-track position bxt + eps(t) (eps_t: array over cfg.ta, metres)."""
    vs, H, ta, Na = cfg.system.vs, cfg.scene.H, cfg.ta, cfg.Na
    ptx = np.column_stack([vs * ta, np.zeros(Na), H * np.ones(Na)])
    prx = np.zeros([2, Na, 3])
    for i in range(2):
        prx[i, :, 0] = vs * ta - cfg.array.bat[i]
        prx[i, :, 1] = cfg.array.bxt[i] + (eps_t if i == 1 else 0.0)
        prx[i, :, 2] = H
    vrx = np.array([np.full(Na, vs), np.sqrt(vs ** 2 + np.gradient(prx[1, :, 1], ta) ** 2)])
    return PlatformTracks(ptx=ptx, vtx=np.full(Na, vs), prx=prx, vrx=vrx)


def iso_point(cfg, h, x=None):
    """Point at azimuth x (default: the reference point's), height h, on the iso-range of the reference point."""
    r0, H, y0 = cfg.scene.r0, cfg.scene.H, cfg.scene.y0
    x = cfg.scene.x0 if x is None else x
    return np.array([x, np.sqrt(r0 ** 2 - (H - h) ** 2), h]) if y0 > 0 else None


def coeffs(cfg, tr, p, ch=1, fit=False):
    return GetCoeffNu(p, tr.ptx, tr.prx[ch], tr.vtx, tr.vrx[ch], tr.ptx, tr.vtx, cfg.prf, cfg.system.wl, cfg.ta,
                      cfg.sq_tx, cfg.sq_rx[ch], cfg.theta_tx, cfg.theta_rx[ch], return_fit=fit)
