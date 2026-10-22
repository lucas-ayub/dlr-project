"""
Result figures of the report (vector PDFs in ../figures), computed with the sata1d package of this repository.

    python make_figures.py all            # every figure (a few minutes on four cores)
    python make_figures.py coeffs rangehist spectra irf_height amb_height sata_controls sata_realistic \
                           sata_refpoint joint_convergence joint_nrx

Intermediate results are cached in ~/DLR/dlr-project_resultados/report_figures_cache (delete it to recompute).
Every method receives only the system parameters and the DEM; the targets are used only to simulate the echoes.
"""
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.normpath(os.path.join(HERE, "..", "..", "sata1d_code"))
sys.path[:0] = [SC, os.path.join(SC, "joint_cases")]
import sata1d as S                                                     # noqa: E402
from sata1d.config import ArrayGeometry                                # noqa: E402
from sata1d.joint_line import height_table, line_maps, channel_maps, reconstruct_joint_line   # noqa: E402
from sweep_geometry_plots import iso_range_dy                          # noqa: E402
import audit_dem as A                                                  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))
_mac = os.path.expanduser("~/mnt/DLR")                                 # the Cowork shell mounts ~/DLR there
CACHE = os.path.join(_mac if os.path.isdir(_mac) else os.path.expanduser("~/DLR"), "dlr-project_resultados", "report_figures_cache")
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True,
                     "grid.color": "0.88", "grid.linewidth": 0.6, "lines.linewidth": 1.1, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.03, "legend.framealpha": 0.9})
W_FULL = 6.3                                                           # text width of the report [in]
COL = dict(ref="#2ca02c", none="#d62728", whole="#ff7f0e", sub="#8c564b", joint_occ="#1f77b4", joint_line="#9467bd")
LAB = dict(ref="reference", none="no SATA", whole="SATA whole band", sub="SATA per sub-band",
           joint_occ="joint (sub-apertures)", joint_line="joint (whole line)")
TERRAIN_C, FILL_C = "#6D4C2F", "#D7C9A7"


def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    r = fn()
    with open(p, "wb") as f:
        pickle.dump(r, f)
    return r


def legend_above(ax, ncol=1, **kw):
    """Legend outside the axes, just above them, so that it never covers the data."""
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol, frameon=False, borderaxespad=0.0,
                    handlelength=1.6, columnspacing=1.2, fontsize=kw.pop("fontsize", 6.4), title_fontsize=6.6, **kw)
    leg._legend_box.align = "left"
    return leg


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def coeffs(cfg, tr, p, i, fit=False):
    return S.GetCoeffNu(p, tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, cfg.system.wl,
                        cfg.ta, cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i], return_fit=fit)


def config(targets, bxt, amps=None):
    cfg = S.make_config(list(targets), nrx=len(bxt), bxt_max=0.0, amplitudes=amps, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    return cfg, S.build_platform_tracks(cfg)


def point(cfg, x, h):
    return cfg.scene.ptg + np.array([x, iso_range_dy(np.array([h]))[0], h])


def tan_theta(cfg):
    return cfg.scene.y0 / cfg.scene.H


BXT4 = A.baseline("N4b300")[0]                                         # [0, 300, 119.8, 100.5] m


# --------------------------------------------------------------------------------------------- coefficients
def fig_coeffs():
    def run():
        bx = np.arange(0.0, 501.0, 25.0)
        ref = np.zeros((3, len(bx), 4))                                 # channel 1..3 (b_at = 1, 2, 3 DPCA steps)
        for k, b in enumerate(bx):
            cfg, tr = config([(0.0, 0.0)], [0.0, b, b, b])
            for i in (1, 2, 3):
                ref[i - 1, k] = coeffs(cfg, tr, cfg.scene.ptg, i)
        hs = np.arange(0.0, 3001.0, 150.0)
        res = {}
        for b in (100.0, 300.0, 500.0):
            cfg, tr = config([(0.0, 0.0)], [0.0, b])
            c0 = np.array(coeffs(cfg, tr, cfg.scene.ptg, 1))
            res[b] = np.array([np.array(coeffs(cfg, tr, point(cfg, 0.0, h), 1)) - c0 for h in hs])
        Tap = S.config.integration_time(cfg.system, cfg.scene)
        return dict(bx=bx, ref=ref, hs=hs, res=res, bat=cfg.array.bat, r0=cfg.scene.r0, tt=tan_theta(cfg), T=Tap)
    d = cached("coeffs", run)
    # reference point
    fig, ax = plt.subplots(1, 4, figsize=(W_FULL, 2.15))
    lab = [r"Rx$_1$ ($b_{at}$ = 7.7 m)", r"Rx$_2$ ($b_{at}$ = 15.4 m)", r"Rx$_3$ ($b_{at}$ = 23.1 m)"]
    sty = ["-", "--", ":"]
    for q, (ylab, scale) in enumerate([(r"$C_0$ [m]", 1.0), (r"$C_1$ [$\mu$s]", 1e6), (r"$C_2$ [$10^{-7}$ s$^2$/m]", 1e7), (r"$\Delta t$ [ms]", 1e3)]):
        for i in range(3):
            ax[q].plot(d["bx"], scale * d["ref"][i, :, q], sty[i], color="k", lw=1.0, label=lab[i])
        ax[q].set_xlabel(r"$b_{xt}$ [m]"); ax[q].set_ylabel(ylab, labelpad=1)
    ax[3].set_ylim(0, 1.8)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.9))
    save(fig, "coeffs_reference_point")
    # residual against height
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    for b, c in zip((100.0, 300.0, 500.0), ("#1f77b4", "#ff7f0e", "#d62728")):
        r = d["res"][b]
        ax[0].plot(d["hs"], r[:, 0], "o", ms=3, color=c, label=fr"exact, $b_{{xt}}$ = {b:g} m")
        ax[0].plot(d["hs"], -b * d["hs"] / (d["r0"] * d["tt"]), "-", color=c, lw=0.9)
    ax[0].plot([], [], "-", color="k", lw=0.9, label=r"$-b_{xt}\,h/(r_0\tan\theta)$")
    ax[0].set_xlabel("height of the scatterer $h$ [m]"); ax[0].set_ylabel(r"$\Delta C_0$ [m]")
    ax[0].legend(loc="lower left", fontsize=6.5)
    r = d["res"][300.0]; wl, fe = 0.25, 1234.8 / 2                   # phase of each term at the edge of the processed band
    terms = [(2 * np.pi * np.abs(r[:, 0]) / wl, r"$2\pi|\Delta C_0|/\lambda$", "-"),
             (2 * np.pi * np.abs(r[:, 2]) * wl * fe ** 2, r"$2\pi|\Delta C_2|\,\lambda f^2$", "-."),
             (2 * np.pi * np.abs(r[:, 1]) * fe, r"$2\pi|\Delta C_1|\,f$", "--")]
    for y, l, st in terms:
        ax[1].semilogy(d["hs"][1:], y[1:], st, color="k", label=l)
    ax[1].axhline(np.pi / 4, color="0.5", lw=0.8)
    ax[1].text(80, np.pi / 4 * 1.5, r"$\pi/4$", fontsize=7, color="0.4")
    ax[1].set_xlabel("height of the scatterer $h$ [m]"); ax[1].set_ylabel(r"phase at $f=B_a/2$ [rad]")
    ax[1].legend(loc="center right", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "coeffs_residual_height")


# --------------------------------------------------------------------------------------------- geometry model
def fig_rangehist():
    def run():
        cfg, tr = config([(0.0, 500.0)], BXT4)
        p = cfg.scene.points[1]
        out = []
        for i in range(4):
            c = coeffs(cfg, tr, p, i, fit=True)
            out.append((c[4]["tau"], c[4]["d_range"], c[4]["coeffs"]))
        return dict(out=out, bxt=cfg.array.bxt)
    d = cached("rangehist", run)
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, 2.6))
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for i, (tau, dr, c) in enumerate(d["out"]):
        fit = c[0] + c[1] * tau + c[2] * tau ** 2
        k = slice(None, None, max(1, len(tau) // 40))
        ax[0].plot(tau[k], 1e3 * (dr[k] - c[0]), "o", ms=2.5, color=cols[i])
        ax[0].plot(tau, 1e3 * (fit - c[0]), "-", color=cols[i], lw=0.9,
                   label=fr"Rx$_{i}$, $b_{{xt}}$ = {d['bxt'][i]:g} m (offset {c[0]:.2f} m)")
        ax[1].plot(tau, 1e3 * (dr - fit), "-", color=cols[i], lw=0.9)
    ax[0].set_xlabel(r"time from closest approach $\tau$ [s]"); ax[0].set_ylabel("bistatic $-$ monostatic range\nminus its offset [mm]")
    legend_above(ax[0], ncol=2, fontsize=5.6)
    ax[1].set_xlabel(r"time from closest approach $\tau$ [s]"); ax[1].set_ylabel("fit error [mm]")
    fig.tight_layout(w_pad=1.2)
    save(fig, "range_history_fit")


# --------------------------------------------------------------------------------------------- reconstruction
def _irf(cfg, tr, s):
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    return np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))


def fig_spectra():
    def run():
        cfg, tr = config([(0.0, 0.0)], BXT4)
        ch = S.generate_channels(cfg, tr)
        rec = S.reconstruct(cfg, tr, ch)
        ref = S.generate_reference(cfg, tr)
        f_ch = np.fft.fftshift(np.fft.fftfreq(cfg.Na_ch, 1 / cfg.PRF_op))
        f = np.fft.fftshift(np.fft.fftfreq(cfg.Na, 1 / cfg.prf))
        sp = lambda s: np.abs(np.fft.fftshift(np.fft.fft(s)))
        return dict(f_ch=f_ch, chs=[sp(c) for c in ch], f=f, rec=sp(rec), ref=sp(ref), prfop=cfg.PRF_op, Ba=cfg.abw)
    d = cached("spectra", run)
    db = lambda a, m: 20 * np.log10(a / m + 1e-9)
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, 2.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
    m = max(c.max() for c in d["chs"])
    pw = np.convolve(d["chs"][1] ** 2, np.ones(25) / 25, mode="same")
    ax[0].plot(d["f_ch"], 10 * np.log10(pw / pw.max()), lw=1.0, color="#1f77b4", label=r"Rx$_1$ (aliased), power over 25 bins")
    ax[0].set_xlim(-d["prfop"] / 2, d["prfop"] / 2); ax[0].set_ylim(-12, 2)
    ax[0].set_xlabel("Doppler frequency of the channel [Hz]"); ax[0].set_ylabel("[dB]")
    ax[0].legend(fontsize=6.2, loc="lower center", framealpha=1)
    m = d["ref"].max()
    ax[1].plot(d["f"], db(d["ref"], m), color=COL["ref"], lw=1.8, label=LAB["ref"])
    ax[1].plot(d["f"], db(d["rec"], m), color="k", lw=0.7, ls="--", label="standard reconstruction (4 channels)")
    ax[1].text(0, -36, r"processed band $B_a$", ha="center", fontsize=6.5, color="0.35")
    for k in range(-2, 3):
        ax[1].axvline(k * d["prfop"], color="0.6", lw=0.6, ls=":")
    ax[1].axvspan(-d["Ba"] / 2, d["Ba"] / 2, color="0.92", zorder=0)
    ax[1].set_xlim(-1000, 1000); ax[1].set_ylim(-40, 3)
    ax[1].set_xlabel("Doppler frequency [Hz]"); ax[1].set_ylabel("[dB]")
    ax[1].legend(fontsize=6.5, loc="lower center")
    fig.tight_layout(w_pad=1.0)
    save(fig, "spectra_reconstruction")


def fig_irf_height():
    def run():
        out = {}
        for h in (0.0, 500.0):
            cfg, tr = config([(0.0, h)], BXT4)
            ch = S.generate_channels(cfg, tr)
            ref = _irf(cfg, tr, S.generate_reference(cfg, tr)); pk = ref.max()
            out[h] = dict(ref=20 * np.log10(ref / pk + 1e-12), none=20 * np.log10(_irf(cfg, tr, S.reconstruct(cfg, tr, ch)) / pk + 1e-12))
        t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
        dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
        return dict(out=out, t=t, dt=dt)
    d = cached("irf_height", run)
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, 2.3), sharey=True)
    for a, h in zip(ax, (0.0, 500.0)):
        a.plot(d["t"], d["out"][h]["none"], color=COL["none"], lw=0.8, label=LAB["none"])
        a.plot(d["t"], d["out"][h]["ref"], color=COL["ref"], lw=1.0, label=LAB["ref"])
        for m in (-2, -1, 1, 2):
            a.axvline(m * d["dt"], color="0.55", lw=0.6, ls=":")
        a.set_xlim(-2.2, 2.2); a.set_ylim(-90, 3); a.set_xlabel("azimuth time [s]")
        a.set_title(f"one target at h = {h:g} m", fontsize=7.5)
    ax[0].set_ylabel("[dB]")
    h_, l_ = ax[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.9))
    save(fig, "irf_flat_vs_height")


def fig_amb_height():
    def run():
        hs = np.array([0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000], float)
        res = {}
        for b in (100.0, 300.0, 500.0):
            bxt = A.baseline(f"N4b{b:g}")[0]
            v = []
            for h in hs:
                cfg, tr = config([(0.0, h)], bxt)
                ch = S.generate_channels(cfg, tr)
                ref = _irf(cfg, tr, S.generate_reference(cfg, tr)); pk = ref.max()
                env = _irf(cfg, tr, S.reconstruct(cfg, tr, ch))
                t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
                dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
                v.append(A.ambiguity_level(t, env, pk, np.array([0.0]), cfg.system.vs, dt))
            res[b] = (np.array(v), bxt)
        return dict(hs=hs, res=res)
    d = cached("amb_height_v2", run)
    fig, ax = plt.subplots(figsize=(0.62 * W_FULL, 2.3))
    for b, c in zip((100.0, 300.0, 500.0), ("#1f77b4", "#ff7f0e", "#d62728")):
        v, bxt = d["res"][b]
        ax.semilogx(d["hs"][1:], v[1:], "o-", ms=2.5, color=c, label=r"$b_{xt}$ = [" + ", ".join(f"{x:g}" for x in bxt) + "] m")
    ax.set_xlabel("height of the target $h$ [m]"); ax.set_ylabel("highest ambiguity [dB]")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "ambiguity_vs_height")


# --------------------------------------------------------------------------------------------- scene helpers
def run_audit(sc, tag):
    def run():
        res, d = A.compute(sc, tag)
        c = d["c"]
        db = {k: 20 * np.log10(v / d["pk"] + 1e-12) for k, v in d["env"].items()}
        dt = d["cfg"].PRF_op * d["cfg"].system.wl * d["cfg"].scene.r0 / (2.0 * d["cfg"].system.vs ** 2)
        return dict(res=res, db=db, t=d["t"], xs=c["xs"], hs=c["hs"], terrain=None, label=c["label"],
                    bxt=d["cfg"].array.bxt, dt=dt, tgrid=np.linspace(-23500, 23500, 4701),
                    hgrid=c["terrain"](np.linspace(-23500, 23500, 4701)))
    return cached(f"audit_{sc}__{tag}", run)


def geometry_panels(ax2, ax3, r, zoom_pad=1500.0, folds=None):
    xs, hs, tg, hg = r["xs"], r["hs"], r["tgrid"], r["hgrid"]
    lo, hi = min(xs.min(), 0.0) - zoom_pad, max(xs.max(), 0.0) + zoom_pad
    k = (tg >= lo) & (tg <= hi)
    ax2.fill_between(tg[k] / 1e3, min(0.0, hg.min()) - 30, hg[k], color=FILL_C, alpha=0.55, lw=0)
    ax2.plot(tg[k] / 1e3, hg[k], color=TERRAIN_C, lw=1.2)
    ax2.scatter(xs / 1e3, hs, s=14, color="#1F4E79", edgecolor="k", lw=0.4, zorder=3, label="point targets")
    ax2.plot([0], [0], "*", ms=8, color="gold", mec="k", mew=0.5, zorder=4, label="reference point")
    if folds is not None:
        for x0 in folds:
            ax2.axvline(x0 / 1e3, color="0.5", ls=":", lw=0.7)
    ax2.set_xlim(lo / 1e3, hi / 1e3); ax2.set_xlabel("azimuth position $x$ [km]"); ax2.set_ylabel("height [m]")
    legend_above(ax2, ncol=2)
    dy = iso_range_dy(hg)
    ax3.plot(tg / 1e3, dy / 1e3, hg, color=TERRAIN_C, lw=0.9)
    ax3.scatter(xs / 1e3, iso_range_dy(hs) / 1e3, hs, s=8, color="#1F4E79", depthshade=False)
    for x, y, h in zip(xs / 1e3, iso_range_dy(hs) / 1e3, hs):
        ax3.plot([x, x], [y, y], [0, h], color="0.6", lw=0.5)
    ax3.scatter([0], [0], [0], marker="*", s=40, color="gold", edgecolor="k", depthshade=False)
    ax3.set_xlabel("azimuth $x$ [km]", labelpad=-6, fontsize=6.5); ax3.set_ylabel(r"ground range $\Delta y$ [km]", labelpad=-6, fontsize=6.5)
    ax3.set_zlabel("height [m]", labelpad=-6, fontsize=6.5)
    ax3.tick_params(labelsize=5.5, pad=-2)
    ax3.view_init(elev=24, azim=-60)
    ax3.set_box_aspect((2.4, 1.0, 0.8), zoom=1.3)


def irf_panel(a, r, keys, xlim=None, title=None):
    for k in [k for k in keys if k != "ref"] + (["ref"] if "ref" in keys else []):
        a.plot(r["t"], r["db"][k], color=COL[k], lw=0.9 if k == "ref" else 0.7,
               label=f"{LAB[k]}: highest ambiguity {r['res'][k]:.1f} dB")
    a.set_ylim(-80, 3); a.set_yticks([-75, -50, -25, 0])
    if xlim:
        a.set_xlim(*xlim)
    a.set_xlabel("azimuth time [s]"); a.set_ylabel("[dB]")
    legend_above(a, title=title)


def bxt_str(b):
    return "[" + ", ".join(f"{x:g}" for x in b) + "] m"


# --------------------------------------------------------------------------------------------- SATA figures
def fig_sata_controls():
    cases = [("ctrl_periodX_5", "N4b100"), ("ctrl_period1p5X_5", "N4b100")]
    rs = [run_audit(*c) for c in cases]
    fig = plt.figure(figsize=(W_FULL, 8.6))
    gs = GridSpec(5, 2, height_ratios=[0.9, 1, 1, 1, 1], hspace=0.95, wspace=0.28, figure=fig)
    for j, r in enumerate(rs):
        a = fig.add_subplot(gs[0, j])
        x0 = r["xs"][2]
        folds = [x0 - 6228.7, x0, x0 + 6228.7]
        k = np.abs(r["tgrid"]) <= 12000
        a.fill_between(r["tgrid"][k] / 1e3, -10, r["hgrid"][k], color=FILL_C, alpha=0.55, lw=0)
        a.plot(r["tgrid"][k] / 1e3, r["hgrid"][k], color=TERRAIN_C, lw=1.1)
        a.scatter(r["xs"] / 1e3, r["hs"], s=12, color="#1F4E79", edgecolor="k", lw=0.4, zorder=3)
        for f in folds:
            a.plot([f / 1e3], [np.interp(f, r["tgrid"], r["hgrid"])], "o", ms=5, mfc="none", mec=COL["none"], mew=1.0)
            a.axvline(f / 1e3, color="0.6", ls=":", lw=0.6)
        a.set_xlim(-12, 12); a.set_ylim(-10, 215)
        a.set_xlabel("azimuth position $x$ [km]"); a.set_ylabel("height [m]")
        a.set_title(["period 6.23 km: same height at $x$ and $x\\pm6.23$ km",
                     "period 9.34 km: different heights at $x$ and $x\\pm6.23$ km"][j], fontsize=7.2, pad=4)
        for q, k in enumerate(["none", "whole", "sub", "joint_line"]):
            irf_panel(fig.add_subplot(gs[1 + q, j]), r, [k, "ref"], (-2.6, 2.6))
    save(fig, "sata_controls")


def fig_sata_realistic():
    r = run_audit("few_mountains_5", "N4b300")
    fig = plt.figure(figsize=(W_FULL, 6.0))
    gs = GridSpec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.8, wspace=0.28, figure=fig)
    geometry_panels(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], projection="3d"), r)
    irf_panel(fig.add_subplot(gs[1, 0]), r, ["ref", "none"])
    irf_panel(fig.add_subplot(gs[1, 1]), r, ["ref", "whole"])
    irf_panel(fig.add_subplot(gs[2, 0]), r, ["ref", "sub"])
    irf_panel(fig.add_subplot(gs[2, 1]), r, ["ref", "joint_line"])
    print("few_mountains_5 N4b300", {k: round(v, 1) for k, v in r["res"].items() if k in LAB}, r["bxt"], r["label"])
    save(fig, "sata_vs_joint_mountains")


def fig_sata_refpoint():
    r = run_audit("one_refpoint", "N4b100")
    fig = plt.figure(figsize=(W_FULL, 4.3))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.85, wspace=0.28, figure=fig)
    geometry_panels(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], projection="3d"), r, zoom_pad=8000.0)
    irf_panel(fig.add_subplot(gs[1, 0]), r, ["ref", "none", "whole"], (-2.6, 2.6))
    irf_panel(fig.add_subplot(gs[1, 1]), r, ["ref", "sub", "joint_line"], (-2.6, 2.6))
    print("one_refpoint N4b100", {k: round(v, 1) for k, v in r["res"].items() if k in LAB}, r["bxt"], r["label"])
    save(fig, "sata_refpoint")


# --------------------------------------------------------------------------------------------- joint figures
def mountain_example():
    """The example of the guides: 5 targets on a 1000 m Gaussian mountain, b_xt = [0, 134.9, 20.5, 8.3] m."""
    terrain = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
    xs = np.array([-5200.0, -2600.0, 0.0, 2600.0, 5200.0])
    cfg, tr = config(list(zip(xs, terrain(xs))), [0.0, 134.9, 20.5, 8.3])
    return cfg, tr, terrain, xs


def fig_joint_convergence():
    def run():
        cfg, tr, terrain, xs = mountain_example()
        ch = S.generate_channels(cfg, tr)
        half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
        hg = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
        tab = height_table(cfg, tr, hg.min(), hg.max())
        M = line_maps(cfg, tr, terrain, tab)
        ref = _irf(cfg, tr, S.generate_reference(cfg, tr)); pk = ref.max()
        t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
        dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
        lev = lambda env: A.ambiguity_level(t, env, pk, xs, cfg.system.vs, dt)
        its = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
        one, une = [], []
        for n in its:
            y, info = reconstruct_joint_line(cfg, tr, ch, M, passes=1, iterations=n, tol=0.0, return_info=True)
            one.append(lev(_irf(cfg, tr, y))); une.append(20 * np.log10(info["unexplained"]))
        two = []
        for n in its:
            y, info = reconstruct_joint_line(cfg, tr, ch, M, passes=2, iterations=n, tol=0.0, return_info=True)
            two.append(lev(_irf(cfg, tr, y)))
        none = lev(_irf(cfg, tr, S.reconstruct(cfg, tr, ch)))
        return dict(its=its, one=one, two=two, une=une, none=none, ref=lev(ref))
    d = cached("joint_convergence", run)
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, 2.3))
    ax[0].semilogx(d["its"], d["one"], "o-", ms=3, color=COL["joint_line"], label="joint (whole line), 1 pass")
    ax[0].semilogx(2 * np.array(d["its"]), d["two"], "s--", ms=3, color="k", lw=0.9, label="joint (whole line), 2 passes")
    ax[0].axhline(d["ref"], color=COL["ref"], lw=1.4, label=f"reference ({d['ref']:.1f} dB)")
    ax[0].axhline(d["none"], color=COL["none"], lw=1.0, ls="--", label=f"no SATA ({d['none']:.1f} dB)")
    ax[0].set_xlabel("conjugate-gradient iterations (all passes)"); ax[0].set_ylabel("highest ambiguity [dB]")
    ax[0].legend(fontsize=6.2, loc="center right")
    ax[1].semilogx(d["its"], d["une"], "o-", ms=3, color="k")
    ax[1].set_xlabel("conjugate-gradient iterations (1 pass)"); ax[1].set_ylabel("unexplained part [dB]")
    fig.tight_layout(w_pad=1.2)
    save(fig, "joint_convergence")


def fig_joint_nrx():
    tags = ["N2b300", "N3b300", "N4b300", "N8b300"]
    rs = [run_audit("many_mountains_49", t) for t in tags]
    fig = plt.figure(figsize=(W_FULL, 6.4))
    gs = GridSpec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.95, wspace=0.28, figure=fig)
    geometry_panels(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], projection="3d"), rs[0], zoom_pad=1500.0)
    for j, r in enumerate(rs):
        a = fig.add_subplot(gs[1 + j // 2, j % 2])
        irf_panel(a, r, ["ref", "joint_line"], (-3.2, 3.2),
                  title=fr"$N_{{rx}}$ = {len(r['bxt'])}, $b_{{xt}}$ = {bxt_str(r['bxt'])}")
        print(tags[j], {k: round(v, 1) for k, v in r["res"].items() if k in LAB})
    save(fig, "joint_nrx_mountains49")


FIGS = dict(coeffs=fig_coeffs, rangehist=fig_rangehist, spectra=fig_spectra, irf_height=fig_irf_height,
            amb_height=fig_amb_height, sata_controls=fig_sata_controls, sata_realistic=fig_sata_realistic,
            sata_refpoint=fig_sata_refpoint, joint_convergence=fig_joint_convergence, joint_nrx=fig_joint_nrx)

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for n in (FIGS if names == ["all"] else names):
        FIGS[n]()
