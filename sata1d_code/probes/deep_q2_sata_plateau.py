"""
Q2, part 2: why the level after SATA levels off for targets 200 m apart with large height steps.
Ramp N = 5, 200 m apart, h_j = j * 2600 tan(alpha): for the central target and channel 0,
the sub-aperture spectrum |STFT|^2 (energy per bin), the phase the map applies to each bin
(read at the cell the bin points to) and the phase the target needs (constant).
The effective gain of the correction is g = sum_b E_b exp(j(applied_b - needed)) / sum_b E_b.
Writes probes/deep_q2_plateau.csv (per alpha) and probes/deep_q2_plateau_bins.npz (bins of one case).
"""
import csv
import numpy as np
import sata1d as S
import sata1d.sata as SS
import run_geometry_sweep as G

WL = 0.25
ALPHAS = [1, 3, 5, 10, 20, 30, 50]


def case(alpha, N=5, step=2600.0, spacing=200.0, ch=0):
    heights = G.heights_from_slopes([alpha] * (N - 1), step)
    xs = (np.arange(N) - (N - 1) / 2.0) * spacing
    cfg = S.make_config(list(zip(xs, heights)), bxt_max=G.BXT_MAX, seed=G.BXT_SEED); tr = S.build_platform_tracks(cfg)
    j = N // 2; p = cfg.scene.points[1 + j]
    dC0 = S.residual_C0(cfg, tr, p, ch)
    mp = SS.build_delta_C0_array(cfg, tr, ch)
    prf, v, r = cfg.PRF_op, cfg.system.vs, cfg.scene.r0
    Tsub, hop, Nzp = SS.subaperture_sizing(r, prf, v, WL, G.SATA_OSF)
    fsub, azpos = SS.subaperture_axis(prf, Nzp, v, WL, r, 0.0)
    win = SS.triangular_window(Tsub)
    pix = SS.az_pixel_of_scatterer(cfg, xs[j], ch)
    line = S.generate_channels(cfg, tr, ptgs=[p])[ch]           # this target alone
    # sub-aperture centred on the target
    start = int(round(pix - 0.5 * Tsub))
    buf = np.zeros(Nzp, complex); buf[:Tsub] = line[start:start + Tsub] * win
    spec = np.fft.fft(buf); E = np.abs(spec) ** 2; E /= E.sum()
    centre = start + 0.5 * Tsub
    pos = np.clip(np.round(azpos + centre).astype(int), 0, cfg.Na_ch - 1)
    applied = 2 * np.pi * mp[pos] / WL            # phase the kernel applies (up to sign convention)
    needed = 2 * np.pi * dC0 / WL
    g = np.sum(E * np.exp(1j * (applied - needed)))
    dh = step * np.tan(np.radians(alpha))
    o = np.argsort(fsub)
    return dict(alpha=alpha, dh=dh, phi_step_deg=np.degrees(2 * np.pi * (S.residual_C0(cfg, tr, cfg.scene.points[2 + j], ch) - dC0) / WL),
                g_abs=abs(g), E_within_1rad=float(E[np.abs(np.angle(np.exp(1j * (applied - needed)))) < 1.0].sum()),
                E_main_lobe=float(E[np.abs(azpos) <= 40].sum()),
                E_own_cells=float(E[np.abs(azpos) <= 6.5].sum())), \
        dict(f=fsub[o], cells=azpos[o], E=E[o], applied_deg=np.degrees(applied[o]), needed_deg=np.degrees(needed), map_cells=mp)


if __name__ == "__main__":
    rows = []
    for a in ALPHAS:
        r, bins = case(a)
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if a == 20:
            np.savez("probes/deep_q2_plateau_bins.npz", **bins)
    with open("probes/deep_q2_plateau.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
