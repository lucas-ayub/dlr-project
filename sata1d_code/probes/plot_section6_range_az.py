"""Range x azimuth maps of delta C0 (channel 0 unless stated) for a synthetic terrain: a hill.
For every range line the reference point is the flat point of that line (as a 2-D processor would build its filter)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
sys.path.insert(0, "."); sys.path.insert(0, "probes")
import sata1d as S, sata1d.sata as SS
from sata1d.reconstruction import GetCoeffNu
import run_geometry_sweep as G

OUT = "section6_figs"; os.makedirs(OUT, exist_ok=True)
cfg = S.make_config([(0.0, 500.0)], bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
H, x0, y0, r0 = cfg.scene.H, cfg.scene.x0, cfg.scene.ptg[1], cfg.scene.r0
ds = cfg.system.vs / cfg.PRF_op
common = lambda i: (tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, cfg.system.wl, cfg.ta,
                    cfg.sq_tx, cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
def c0(p, i): return GetCoeffNu(np.asarray(p, float), *common(i))[0]

# terrain: a hill, 1000 m high, centred 3 km beyond the swath centre in ground range
def terrain(x, y): return 1000.0 * np.exp(-0.5 * ((x / 2500.0) ** 2 + ((y - y0 - 3000.0) / 6000.0) ** 2))
ys = y0 + np.linspace(-15000, 15000, 31)                  # ground range of each range line
rs = np.sqrt(ys ** 2 + H ** 2)                             # slant range of each line
xs = x0 + np.linspace(-10000, 10000, 81)                   # azimuth positions [m]
th = np.degrees(np.arctan2(ys, H))

# ---- needed delta C0 on the (range, azimuth) grid, four channels
need = np.zeros((4, len(ys), len(xs)))
for a, (y, r) in enumerate(zip(ys, rs)):
    refs = [c0([x0, y, 0.0], i) for i in range(4)]
    for b, x in enumerate(xs):
        h = terrain(x - x0, y)
        p = [x, np.sqrt(r ** 2 - (H - h) ** 2), h]
        for i in range(4): need[i, a, b] = c0(p, i) - refs[i]
    print(a, flush=True)
hgrid = np.array([[terrain(x - x0, y) for x in xs] for y in ys])
np.save(f"{OUT}/range_az.npy", dict(xs=xs, ys=ys, rs=rs, need=need, h=hgrid))

ext = [(xs[0] - x0) / 1e3, (xs[-1] - x0) / 1e3, (rs[0] - r0) / 1e3, (rs[-1] - r0) / 1e3]
def show(ax, Z, title, cmap, vmin, vmax, label):
    im = ax.imshow(Z, aspect="auto", origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10); cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(label); return im

fig, axs = plt.subplots(2, 3, figsize=(17, 9)); fig.subplots_adjust(wspace=0.35, hspace=0.32)
show(axs[0, 0], hgrid, "terreno h(range, azimute)", "terrain", 0, 1000, "h [m]")
show(axs[0, 1], need[0] * 1e3, "$\\Delta C_0$ necessário, canal 0", "Spectral", -240, 0, "$\\Delta C_0$ [mm]")
ratio = np.where(hgrid > 50, need[0] * 1e3 / hgrid, np.nan)
show(axs[0, 2], ratio, "$\\Delta C_0 / h$, canal 0 (só onde h > 50 m)", "viridis", np.nanmin(ratio), np.nanmax(ratio), "[mm por m de altura]")
for i, ax in zip((1, 2, 3), axs[1]):
    show(ax, need[i] * 1e3, f"$\\Delta C_0$ necessário, canal {i} ($b_{{xt}}$ = {tr.prx[i][0][1] - tr.ptx[0][1]:.1f} m)", "Spectral", -240, 0, "$\\Delta C_0$ [mm]")
for ax in axs.flat: ax.set_xlabel("azimute [km]"); ax.set_ylabel("slant range $-\\,r_0$ [km]")
axs[0, 0].text(-9.5, 4.6, f"ângulo de visada {th[0]:.1f}° (baixo) a {th[-1]:.1f}° (alto)", fontsize=8.5, color="w")
fig.savefig(f"{OUT}/s6_rangeaz_needed.png", dpi=115, bbox_inches="tight"); plt.close(fig)

# ---- what each map writes, line by line (channel 0): targets every 400 m on the terrain
from sata1d.subband import subband_frequency_beam
cells = np.arange(cfg.Na_ch); xcell = (cells - cfg.Na_ch / 2) * ds - x0
xt = x0 + np.arange(-8000, 8001, 400.0)
hw = SS.sata_footprint_halfwidth(cfg)
offs = {"whole band": SS.sata_image_offsets(cfg, 0.0)}
for k in range(4): offs[f"sub-banda {k}"] = SS.sata_image_offsets(cfg, subband_frequency_beam(cfg, k)[0])
maps = {n: np.zeros((len(ys), cfg.Na_ch)) for n in list(offs) + ["look-up table"]}
for a, (y, r) in enumerate(zip(ys, rs)):
    ref = c0([x0, y, 0.0], 0)
    by_pix = {}
    for x in xt:
        h = terrain(x - x0, y); p = [x, np.sqrt(r ** 2 - (H - h) ** 2), h]
        by_pix[SS.az_pixel_of_scatterer(cfg, x - x0, 0)] = [c0(p, 0) - ref]
    for n, o in offs.items(): maps[n][a] = SS.footprint_map(by_pix, cfg.Na_ch, hw, o)
    maps["look-up table"][a] = np.interp(xcell, xs - x0, need[0, a])
    print("map", a, flush=True)
np.save(f"{OUT}/range_az_maps.npy", dict(maps=maps, xcell=xcell, rs=rs))

ext2 = [-22, 22, (rs[0] - r0) / 1e3, (rs[-1] - r0) / 1e3]
sel = np.abs(xcell) <= 22000
fig, axs = plt.subplots(2, 3, figsize=(17, 9)); fig.subplots_adjust(wspace=0.3, hspace=0.32)
order = ["whole band", "sub-banda 0", "sub-banda 1", "sub-banda 2", "sub-banda 3", "look-up table"]
for ax, n in zip(axs.flat, order):
    Z = maps[n][:, sel] * 1e3
    im = ax.imshow(Z, aspect="auto", origin="lower", extent=ext2, cmap="Spectral", vmin=-240, vmax=0, interpolation="nearest")
    ax.set_title(("mapa atual, kernel " + n) if n != "look-up table" else "look-up table (a mesma para todos os kernels)", fontsize=10)
    ax.axvline(-8.4, color="k", lw=0.6, ls=":"); ax.axvline(8.4, color="k", lw=0.6, ls=":")
    ax.set_xlabel("azimute [km]  (pontilhado: extensão dos alvos, ±8 km)"); ax.set_ylabel("slant range $-\\,r_0$ [km]")
cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02); cb.set_label("$\\Delta C_0$ escrito no mapa [mm]")
fig.savefig(f"{OUT}/s6_rangeaz_maps.png", dpi=115, bbox_inches="tight"); plt.close(fig)
print("ok")
