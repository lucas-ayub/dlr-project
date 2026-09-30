"""
Look-up-table version of the delta C0 map: coefficients computed with the FULL range history (GetCoeffNu)
for every azimuth cell of the channel line, the terrain height at each cell taken from the DEM
(linear interpolation between the targets, held constant beyond them), each cell's point placed on the
iso-range surface. The kernels read the table through f -> beta -> x.
Variants (whole band and per sub-band each):
  current      footprint map with fold images; per sub-band values fitted over the band's angles (getcoeff_beam)
  full values  same maps, but per sub-band values from the full history (GetCoeffNu) as well
  LUT          the look-up table above, read directly (no fold images)
Writes probes/check_lut.csv and check_lut/*.png.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S
import sata1d.sata as SS
from sata1d.subband import subband_frequency_beam, sata_1d_subband, build_delta_C0_subband_array
from sata1d.reconstruction import channel_spectrum, reconstruct_from_spectra
import run_geometry_sweep as G

WL = 0.25
T20 = np.tan(np.radians(20))
SCENES = {
    "one target, h = 500 m": ([0.0], [500.0]),
    "ramp, 5 targets 2.6 km apart, h = 0-291 m": ((np.arange(5) - 2) * 2600.0, np.arange(5) * 200 * T20),
    "ramp, 5 targets 200 m apart, h = 0-291 m": ((np.arange(5) - 2) * 200.0, np.arange(5) * 200 * T20),
}


def lut(cfg, tr, i, xs, hs, step=4):
    """delta C0 [m] for every cell of channel i's line, from the full range history."""
    ds = cfg.system.vs / cfg.PRF_op
    cells = np.arange(cfg.Na_ch)
    dx = (cells - cfg.Na_ch / 2) * ds - cfg.scene.x0 - 0.5 * cfg.array.bat[i]       # inverse of az_pixel_of_scatterer
    H, r0, y0 = cfg.scene.H, cfg.scene.r0, cfg.scene.ptg[1]
    coarse = cells[::step]
    val = np.zeros(len(coarse))
    xs = np.asarray(xs, float); hs = np.asarray(hs, float)
    for n, c in enumerate(coarse):
        x = cfg.scene.x0 + dx[c]
        h = np.interp(x, xs + cfg.scene.x0, hs)                  # DEM: linear between targets, held beyond
        if h == 0.0:
            continue
        p = np.array([x, np.sqrt(r0 ** 2 - (H - h) ** 2), h])
        val[n] = S.residual_C0(cfg, tr, p, i)
    return np.interp(cells, coarse, val)


def run(name, xs, hs):
    cfg = S.make_config(list(zip(xs, hs)), bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    ref = F(S.generate_reference(cfg, tr)); pk = ref.max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    amb = lambda rec: G.ambiguity_level(t, F(rec), pk, np.asarray(xs, float), cfg.system.vs, G.ambiguity_times(cfg))
    kw = dict(rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=WL, r=cfg.scene.r0, inverse=True, sata_osf=G.SATA_OSF)
    L = [lut(cfg, tr, i, xs, hs) for i in range(cfg.Nrx)]
    maps = {"current": ([SS.build_delta_C0_array(cfg, tr, i) for i in range(cfg.Nrx)],
                        [[build_delta_C0_subband_array(cfg, tr, i, k) for i in range(cfg.Nrx)] for k in range(cfg.Nrx)]),
            "full values": ([SS.build_delta_C0_array(cfg, tr, i) for i in range(cfg.Nrx)],
                            [[SS.build_delta_C0_array(cfg, tr, i, f_centre=subband_frequency_beam(cfg, k)[0]) for i in range(cfg.Nrx)]
                             for k in range(cfg.Nrx)]),
            "LUT": (L, [L] * cfg.Nrx)}
    out = dict(scene=name, none=amb(S.reconstruct(cfg, tr, ch)), oracle=amb(S.reconstruct(cfg, tr, G.oracle_channels(cfg, tr))))
    for key, (Mw, Mk) in maps.items():
        W = np.array([S.sata_1d(ch[i], Mw[i], **kw) for i in range(cfg.Nrx)])
        out[f"{key}_whole"] = amb(S.reconstruct(cfg, tr, W))
        spec = []
        for k in range(cfg.Nrx):
            fk = subband_frequency_beam(cfg, k)[0]
            C = [sata_1d_subband(ch[i], Mk[k][i], f_k=fk, **kw) for i in range(cfg.Nrx)]
            spec.append([channel_spectrum(cfg, C[q]) for q in range(cfg.Nrx)])
        out[f"{key}_sub"] = amb(reconstruct_from_spectra(cfg, tr, spec))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out, cfg, maps


if __name__ == "__main__":
    os.makedirs("check_lut", exist_ok=True)
    rows, keep = [], {}
    for name, (xs, hs) in SCENES.items():
        o, cfg, maps = run(name, xs, hs); rows.append(o); keep[name] = (cfg, maps, xs, hs)
    with open("probes/check_lut.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.save("check_lut/maps.npy", np.array([{n: (v[1]["current"][0][0], v[1]["current"][1][0][0], v[1]["LUT"][0][0], v[0].Na_ch,
                                                  v[0].system.vs / v[0].PRF_op) for n, v in keep.items()}], dtype=object), allow_pickle=True)
