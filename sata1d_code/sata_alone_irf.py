"""
SATA 1D without folds (PRF 2000 Hz > Doppler bandwidth: no reconstruction needed), seen in the IRF amplitude:
two receivers sampled at the full PRF, receiver A on the transmitter (b_xt = 0) and receiver B with a cross-track
baseline b_xt, and the two focused images added (no reconstruction). With relief the two receivers see every target
with a different phase, so the sum loses amplitude; SATA on each receiver puts them back in phase.
Same terrains, DEM table and SATA calls as part B of verify_sata_alone.py (which is not changed).
Pages of sata_alone_irf.pdf:
    left top     |IRF| [dB] of the sum over the whole azimuth line: reference (monostatic), no SATA, SATA whole band,
                 SATA per sub-band
    left bottom  the same, zoom on every target
    right        the scene: 2D zoom on the targets, 3D over the whole azimuth line
Pages of sata_alone_irf_prats.pdf (as Fig. 4.4 of Prats' thesis): linear amplitude normalised to the peak of the
reference, along-track +-25 m around every target, with also the conventional correction: one phase per receiver
for one height (the mean DEM height over |x| <= 8 km), applied with the same kernel and a flat DEM at that height.

    python sata_alone_irf.py --compute --budget 160     # from sata1d_code, repeat until nothing is missing
    python sata_alone_irf.py --pdf
Output: sata_alone/irf_cache/*.npz, sata_alone/irf_pages/*.png, sata_alone/sata_alone_irf.pdf, sata_alone/sata_alone_irf_prats.pdf
"""
import time, argparse
import os, sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "joint_cases"))
import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
from sata1d.reconstruction import reconstruct_from_spectra, channel_spectrum
from sata1d.subband import sata_channels_subband
import verify_sata_alone as V
from sweep_joint_all import bundle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
from sweep_geometry_plots import scene_coordinates, geometry_2d, geometry_3d, iso_range_dy

OUT = os.path.join(V.OUT, "irf_pages")
CACHE = os.path.join(V.OUT, "irf_cache")
HBAR_HALF = 8000.0                 # the conventional height: mean DEM over |x| <= 8 km
V.TERRAINS["flat1760"] = (lambda x: 0.0 * np.asarray(x, float) + 1760.0, "flat terrain at h = 1760 m (receiver B sees every target about 180 deg off receiver A)",
                          (-5000.0, -2000.0, 0.0, 2500.0, 5500.0))
CASES = [("flat1760", 300.0), ("plateaus", 300.0), ("plateaus", 100.0), ("ramp2", 300.0), ("ramp8", 100.0),
         ("ramp8", 300.0), ("sine", 300.0), ("mountain", 100.0), ("mountain", 300.0)]
STY = {"ref": ("Reference", dict(color="tab:green", lw=2.2)),
       "none": ("no SATA", dict(color="tab:red", lw=1.4, ls=(0, (5, 3)))),
       "whole": ("SATA whole band", dict(color="tab:blue", lw=1.1)),
       "sub": ("SATA per sub-band", dict(color="tab:orange", lw=1.0, ls=(0, (2, 2)))),
       "conv": ("conventional (one phase for the mean DEM height)", dict(color="tab:purple", lw=1.3, ls="-."))}


def one_receiver(terrain, xs, bxt, hbar):
    """One receiver at the full PRF: focused lines (complex) without SATA, with SATA whole band and per sub-band
    (the calls of verify_sata_alone.run_b), and the monostatic reference."""
    cfg = S.make_config([(x, float(terrain(x))) for x in xs], nrx=1, bxt_max=0.0)
    cfg = replace(cfg, array=ArrayGeometry(bat=np.zeros(1), bxt=np.array([float(bxt)])))
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    M = S.terrain_lookup_table(cfg, tr, 0, terrain, step=16) if bxt else np.zeros(cfg.Na)
    out = {"none": S.reconstruct(cfg, tr, ch)}
    corr = SS.sata_1d(ch[0], M, inverse=True, sata_osf=V.OSF, **V.kernel_args(cfg))
    out["whole"] = S.reconstruct(cfg, tr, corr[None])
    corr = sata_channels_subband(cfg, tr, ch, 0, sata_osf=V.OSF, maps=[M])
    out["sub"] = reconstruct_from_spectra(cfg, tr, [[channel_spectrum(cfg, corr[0])]], True)
    Mc = S.terrain_lookup_table(cfg, tr, 0, lambda x: 0.0 * np.asarray(x, float) + hbar, step=16) if bxt else np.zeros(cfg.Na)
    out["conv"] = S.reconstruct(cfg, tr, SS.sata_1d(ch[0], Mc, inverse=True, sata_osf=V.OSF, **V.kernel_args(cfg))[None])
    foc = V.focuser(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]))
    img = {k: foc(v) for k, v in out.items()}
    img["ref"] = foc(V.cut_band(cfg, S.generate_reference(cfg, tr)))
    return cfg, img


def mean_height(terrain):
    return float(np.mean(np.asarray(terrain(np.linspace(-HBAR_HALF, HBAR_HALF, 4001)), float)))


def cache_path(tname, bxt):
    return os.path.join(CACHE, f"{tname}_{bxt:g}.npz")


def compute(tname, bxt):
    """Receiver A (b_xt = 0) and receiver B (b_xt = bxt), focused images added: (A + B) / 2."""
    terrain, desc, xs = V.TERRAINS[tname]
    hbar = mean_height(terrain)
    cfg, a = one_receiver(terrain, xs, 0.0, hbar)
    _, b = one_receiver(terrain, xs, bxt, hbar)
    img = {k: 0.5 * (a[k] + b[k]) for k in ("none", "whole", "sub", "conv")}
    img["ref"] = a["ref"]
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cache_path(tname, bxt), hbar=hbar, **img)


def images(tname, bxt):
    terrain, desc, xs = V.TERRAINS[tname]
    cfg = S.make_config([(x, float(terrain(x))) for x in xs], nrx=1, bxt_max=0.0)
    z = np.load(cache_path(tname, bxt))
    img = {k: z[k] for k in ("ref", "none", "whole", "sub", "conv")}
    return cfg, desc, xs, [float(terrain(x)) for x in xs], img, float(z["hbar"])


def interp_around(I, c, up=16, half=64):
    """Band-limited interpolation (zero-padded FFT) of I[c - half : c + half]; returns offsets [cells] and values."""
    w = np.arange(c - half, c + half)
    F = np.fft.fftshift(np.fft.fft(I[w]))
    n = len(w)
    Z = np.zeros(n * up, complex)
    Z[(n * up - n) // 2:(n * up + n) // 2] = F
    return np.arange(n * up) / up - half, np.fft.ifft(np.fft.ifftshift(Z)) * up


def geometry_panels(fig, gs, cfg, xs, hs, tname):
    """Scene: 2D zoom on the targets and the reference point (top), 3D over the whole azimuth line (bottom)."""
    terrain = V.TERRAINS[tname][0]
    xs_, dys, hs_ = scene_coordinates(np.asarray(xs, float), np.asarray(hs, float))
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    km = FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    xg = np.linspace(-half, half, 9601); hg = np.asarray(terrain(xg), float)
    g2 = fig.add_subplot(gs[0, 1])
    lo2, hi2 = min(xs_.min(), 0.0), max(xs_.max(), 0.0); pad2 = max(0.12 * (hi2 - lo2), 500.0)
    hz = hg[(xg >= lo2 - pad2) & (xg <= hi2 + pad2)]
    geometry_2d(g2, xs_, dys, hs_, False, title=None, connect=False, annotate=True)
    g2.fill_between(xg, min(hg.min(), 0) - 0.05 * max(np.ptp(hg), 1.0), hg, color="#D7C9A7", alpha=0.45, lw=0)
    g2.plot(xg, hg, color="#6D4C2F", lw=1.2, zorder=2, label="terrain (the scene and the DEM)")
    ylo, yhi = min(hz.min(), 0.0), max(hz.max(), hs_.max(), 1.0)
    g2.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    g2.plot(0.0, 0.0, marker="*", ms=13, color="gold", mec="k", ls="none", zorder=6, label="reference point ($x$ = 0, $h$ = 0)")
    g2.set_xlim(lo2 - pad2, hi2 + pad2)
    g2.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    g2.set_title("Scene: zoom on the point targets and the reference point", fontsize=10, pad=34)
    g2.xaxis.set_major_formatter(km); g2.set_xlabel("azimuth position $x$ [km]")
    g3 = fig.add_subplot(gs[1, 1], projection="3d")
    geometry_3d(g3, xs_, dys, hs_, False, connect=False, title=None)
    g3.plot(xg, iso_range_dy(hg), hg, color="#6D4C2F", lw=1.0)
    g3.plot([0.0], [0.0], [0.0], marker="*", ms=10, color="gold", mec="k", ls="none", zorder=6)
    g3.set_xlim(-half, half); g3.xaxis.set_major_formatter(km); g3.set_xlabel("azimuth $x$ [km]", fontsize=8, labelpad=4)
    g3.set_zlim(min(0.0, hg.min()), max(hg.max(), 50.0) * 1.15)
    g3.set_title(f"Scene: whole azimuth line ({2 * half / 1e3:.0f} km) in 3D (x, $\\Delta y$, h)", fontsize=10)
    for t_ in list(g3.texts):
        t_.remove()


def page(tname, bxt, out_png):
    cfg, desc, xs, hs, img, hbar = images(tname, bxt)
    vs = cfg.system.vs
    xa = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf * vs            # azimuth position of every image cell [m]
    pk = np.abs(img["ref"]).max()
    fig = plt.figure(figsize=(21, 13))
    gs = GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], hspace=0.3, wspace=0.12,
                  left=0.045, right=0.985, top=0.885, bottom=0.055, figure=fig)
    geometry_panels(fig, gs, cfg, xs, hs, tname)
    db = {k: 20 * np.log10(np.abs(img[k]) / pk + 1e-12) for k in ("ref", "none", "whole", "sub")}
    n = len(xs); ncol = int(np.ceil(n / 2))
    sub = GridSpecFromSubplotSpec(3, ncol, subplot_spec=gs[:, 0], wspace=0.1, hspace=0.55, height_ratios=[1.15, 1, 1])
    full = fig.add_subplot(sub[0, :])
    for k in ("ref", "whole", "sub", "none"):
        name, st = STY[k]
        full.plot(xa, db[k], label=name, **st)
    full.set_ylim(-70, 5); full.set_xlim(-12000, 12000); full.grid(alpha=0.3)
    full.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:g}"))
    full.set_xlabel("azimuth position [km]", fontsize=9); full.set_ylabel("|IRF| [dB]")
    full.set_title("Whole IRF (sum of the two receivers)", fontsize=10)
    hl = full.get_legend_handles_labels()
    for j, (x, h) in enumerate(zip(xs, hs)):
        z = fig.add_subplot(sub[1 + j // ncol, j % ncol])
        w = np.abs(xa - x) <= 90.0
        for k in ("ref", "whole", "sub", "none"):
            name, st = STY[k]
            z.plot(xa[w] - x, db[k][w], label=name, **dict(st, lw=st["lw"] + (0.8 if k == "none" else 0)))
        z.set_ylim(-50, 3); z.set_xlim(-90, 90); z.grid(alpha=0.3); z.tick_params(labelsize=8)
        z.set_xlabel("along-track distance from the target [m]", fontsize=8)
        p = {k: db[k][w].max() for k in db}
        z.set_title(f"x = {x / 1e3:g} km, h = {h:.0f} m\npeak [dB]: ref {p['ref']:.1f} | no SATA {p['none']:.1f} | SATA {p['whole']:.1f}", fontsize=8)
        if j % ncol:
            z.set_yticklabels([])
        else:
            z.set_ylabel("|IRF| [dB]")
    fig.legend(*hl, loc="upper left", bbox_to_anchor=(0.045, 0.935), ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Two receivers without folds (PRF {cfg.prf:.0f} Hz > Doppler bandwidth, no reconstruction needed), focused images added | "
                 f"receiver A $b_{{xt}}$ = 0, receiver B $b_{{xt}}$ = {bxt:g} m\n"
                 f"{desc[0].upper() + desc[1:]}, {len(xs)} targets | SATA reads the DEM table (terrain_lookup_table)", fontsize=12)
    fig.savefig(out_png, dpi=80); plt.close(fig)


def page_prats(tname, bxt, out_png):
    """As Fig. 4.4 of Prats' thesis: |IRF| linear, normalised to the peak of the reference, +-25 m around every target."""
    cfg, desc, xs, hs, img, hbar = images(tname, bxt)
    vs = cfg.system.vs; ds = vs / cfg.prf
    fig = plt.figure(figsize=(21, 10.5))
    gs = GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.4, 1], hspace=0.36, wspace=0.12,
                  left=0.045, right=0.985, top=0.85, bottom=0.07, figure=fig)
    geometry_panels(fig, gs, cfg, xs, hs, tname)
    n = len(xs); ncol = int(np.ceil(n / 2))
    sub = GridSpecFromSubplotSpec(2, ncol, subplot_spec=gs[:, 0], wspace=0.1, hspace=0.42)
    order = ("ref", "whole", "sub", "conv", "none")
    for j, (x, h) in enumerate(zip(xs, hs)):
        z = fig.add_subplot(sub[j // ncol, j % ncol])
        c = int(round(cfg.Na // 2 + x / ds))
        c = c - 8 + int(np.argmax(np.abs(img["ref"][c - 8:c + 8])))
        off, r = interp_around(img["ref"], c)
        xm = (c + off - cfg.Na // 2) * ds - x
        a0 = np.abs(r).max()
        pk = {}
        for k in order:
            name, st = STY[k]
            _, v = interp_around(img[k], c)
            a = np.abs(v) / a0
            pk[k] = a[np.abs(xm) <= 25].max()
            z.plot(xm, a, label=name, **dict(st, lw=st["lw"] + (0.8 if k == "none" else 0)))
        z.set_xlim(-25, 25); z.set_ylim(0, 1.12); z.grid(alpha=0.3); z.tick_params(labelsize=8)
        z.set_xlabel("along-track [m]", fontsize=8)
        z.set_title(f"x = {x / 1e3:g} km, h = {h:.0f} m\npeak: no SATA {pk['none']:.2f} | conventional {pk['conv']:.2f} | SATA {pk['whole']:.2f}", fontsize=8)
        if j % ncol:
            z.set_yticklabels([])
        else:
            z.set_ylabel("normalised amplitude")
        if j == 0:
            hl = z.get_legend_handles_labels()
    hl = (hl[0], [l.replace("mean DEM height", f"mean DEM height, {hbar:.0f} m") for l in hl[1]])
    fig.legend(*hl, loc="upper left", bbox_to_anchor=(0.045, 0.93), ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f"Two receivers without folds (PRF {cfg.prf:.0f} Hz > Doppler bandwidth, no reconstruction needed), focused images added | "
                 f"receiver A $b_{{xt}}$ = 0, receiver B $b_{{xt}}$ = {bxt:g} m\n"
                 f"{desc[0].upper() + desc[1:]}, {len(xs)} targets", fontsize=12)
    fig.savefig(out_png, dpi=85); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true"); ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--pdf", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    if a.compute:
        for tn, b in CASES:
            if os.path.exists(cache_path(tn, b)):
                continue
            if time.time() - t0 > a.budget:
                print("budget reached"); return
            compute(tn, b); print("computed", tn, b, f"{time.time() - t0:.0f} s", flush=True)
        print("all computed")
    if a.pdf:
        os.makedirs(OUT, exist_ok=True)
        for kind, fn, name in (("db", page, "sata_alone_irf.pdf"), ("prats", page_prats, "sata_alone_irf_prats.pdf")):
            if a.only and a.only != kind:
                continue
            pngs, outline = [], []
            for k, (tn, b) in enumerate(CASES):
                p = os.path.join(OUT, f"{kind}_{k + 1:02d}_{tn}_{b:g}.png")
                fn(tn, b, p)
                outline.append((f"{V.TERRAINS[tn][1]} | receiver B bxt {b:g} m", len(pngs), 0)); pngs.append(p)
                print(p, f"{time.time() - t0:.0f} s", flush=True)
            print("wrote", bundle(pngs, os.path.join(V.OUT, name), outline))


if __name__ == "__main__":
    main()
