"""
Figures for the three open points (sub-band vs whole band, metric/heatmaps, saturation).
Writes to open_points/.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sata1d as S
import sata1d.sata as SS
import run_geometry_sweep as G
import sweep_v3 as V
from sata1d.subband import build_delta_C0_subband_array, subband_axis, subband_frequency_beam
from sata1d.reconstruction import GetCoeffNu

OUT = "open_points"
WL = 0.25
C = dict(ref="#2E7D32", none="#C62828", whole="#1565C0", sub="#EF6C00", flat="#2E7D32", grey="0.55")


def stft_phase_maps(cfg, tr, line, ch):
    prf, v, r = cfg.PRF_op, cfg.system.vs, cfg.scene.r0
    Tsub, hop, Nzp = SS.subaperture_sizing(r, prf, v, WL, G.SATA_OSF)
    win = SS.triangular_window(Tsub)
    _, az_w = SS.subaperture_axis(prf, Nzp, v, WL, r, 0.0)
    map_w = SS.build_delta_C0_array(cfg, tr, ch)
    maps_k = [build_delta_C0_subband_array(cfg, tr, ch, k) for k in range(cfg.Nrx)]
    axes_k = [subband_axis(prf, Nzp, v, WL, r, subband_frequency_beam(cfg, k)[0])[1] for k in range(cfg.Nrx)]
    starts = np.arange(0, len(line) - Tsub, hop)
    mag = np.zeros((len(starts), Nzp)); ph_w = np.zeros_like(mag); ph_k = np.zeros((cfg.Nrx,) + mag.shape)
    for n, s0 in enumerate(starts):
        buf = np.zeros(Nzp, complex); buf[:Tsub] = line[s0:s0 + Tsub] * win
        mag[n] = np.abs(np.fft.fft(buf))
        c = s0 + 0.5 * Tsub
        pw = np.clip(np.round(az_w + c).astype(int), 0, len(line) - 1)
        ph_w[n] = 2 * np.pi / WL * map_w[pw]
        for k in range(cfg.Nrx):
            pk_ = np.clip(np.round(axes_k[k] + c).astype(int), 0, len(line) - 1)
            ph_k[k, n] = 2 * np.pi / WL * maps_k[k][pk_]
    t = (starts + 0.5 * Tsub - cfg.Na_ch / 2) / prf
    f = np.fft.fftfreq(Nzp, 1 / prf)
    o = np.argsort(f)
    return t, f[o], mag[:, o], ph_w[:, o], ph_k[:, :, o]


def fig_tf():
    scenes = [("one target, h = 300 m", [(0.0, 300.0)]),
              ("two targets 2.6 km apart, h = 700 / 300 m", [(-1300.0, 700.0), (1300.0, 300.0)])]
    fig, axs = plt.subplots(len(scenes), 4, figsize=(17, 8.2), constrained_layout=True)
    for row, (name, tg) in zip(axs, scenes):
        cfg = S.make_config(tg); tr = S.build_platform_tracks(cfg)
        line = S.generate_channels(cfg, tr)[0]
        t, f, mag, pw, pk = stft_phase_maps(cfg, tr, line, 0)
        m = 20 * np.log10(mag / mag.max() + 1e-9)
        occ = m > -30
        wrap = lambda x: np.degrees(np.angle(np.exp(1j * x)))
        ext = [f[0], f[-1], t[0], t[-1]]
        im = row[0].imshow(m, aspect="auto", origin="lower", extent=ext, vmin=-40, vmax=0, cmap="Greys")
        row[0].set_title(f"{name}\nchannel 0, sub-aperture spectra |STFT| [dB]", fontsize=9.5)
        fig.colorbar(im, ax=row[0], fraction=0.05)
        for a, ph, lab in ((row[1], pw, "whole band"), (row[2], pk[1], "sub-band k = 1 (-250 ... 0 Hz)")):
            show = np.where(occ, wrap(ph), np.nan)
            im = a.imshow(show, aspect="auto", origin="lower", extent=ext, vmin=-180, vmax=180, cmap="twilight")
            a.set_title(f"correction applied, {lab} [deg]\n(cells with energy above -30 dB)", fontsize=9.5)
            fig.colorbar(im, ax=a, fraction=0.05)
        dmax = np.zeros_like(pw)
        for k in range(cfg.Nrx):
            dmax = np.maximum(dmax, np.abs(wrap(pk[k] - pw)))
        show = np.where(occ, dmax, np.nan)
        im = row[3].imshow(show, aspect="auto", origin="lower", extent=ext, vmin=0, vmax=5, cmap="magma_r")
        en = mag ** 2
        frac = en[occ & (dmax > 1.0)].sum() / en.sum() * 100
        row[3].set_title(f"max over k |sub-band k - whole band| [deg]\nenergy on cells differing by > 1 deg: {frac:.2g} %",
                         fontsize=9.5)
        fig.colorbar(im, ax=row[3], fraction=0.05)
        for a in row:
            a.set_xlabel("baseband Doppler of the bin [Hz]"); a.set_ylabel("sub-aperture centre [s]"); a.set_ylim(-1.5, 1.5)
    fig.savefig(os.path.join(OUT, "tf_correction.png"), dpi=110); plt.close(fig)


def table_lookangle():
    rows = []
    for h in (100, 500, 1000, 3000, 10000):
        cfg = S.make_config([(0.0, h)]); tr = S.build_platform_tracks(cfg); p = cfg.scene.points[1]
        f = np.linspace(-cfg.abw / 2, cfg.abw / 2, 401)
        r = dict(h=h)
        for i in range(cfg.Nrx):
            com = (tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx, cfg.prf, WL, cfg.ta, cfg.sq_tx,
                   cfg.sq_rx[i], cfg.theta_tx, cfg.theta_rx[i])
            d = np.array(GetCoeffNu(p, *com)) - np.array(GetCoeffNu(cfg.scene.ptg, *com))
            ph = 2 * np.pi * ((-d[1] + d[3]) * f + d[2] * WL * f ** 2)
            r[f"c0_{i}"] = np.degrees(2 * np.pi * d[0] / WL); r[f"sp_{i}"] = np.degrees(np.ptp(ph))
        rows.append(r)
    return rows


def band_errors():
    """Error of each output sub-band vs the flat reconstruction, no SATA / whole / sub."""
    scenes = {"1 target, h = 240 m": [(0.0, 240.0)],
              "2 targets 3 km apart, h = 240 / -120 m": [(-1500.0, 240.0), (1500.0, -120.0)],
              "ramp, 9 targets 200 m apart, 15 deg": [(x, (x + 800) * np.tan(np.radians(15))) for x in np.arange(-800, 801, 200.0)],
              "random heights, 5 targets 200 m apart": list(zip((-400.0, -200.0, 0.0, 200.0, 400.0), (0, 20, 48, 96, 43)))}
    out = []
    for name, tg in scenes.items():
        cfg = S.make_config(tg); tr = S.build_platform_tracks(cfg)
        cf0 = S.make_config(tg, bxt_max=0.0); tr0 = S.build_platform_tracks(cf0)
        ch = S.generate_channels(cfg, tr)
        whole_ch = S.sata_channels(cfg, tr, ch, sata_osf=G.SATA_OSF)
        sub_ch = [S.sata_channels_subband(cfg, tr, ch, k, sata_osf=G.SATA_OSF) for k in range(cfg.Nrx)]
        dch = [np.linalg.norm(s - whole_ch) / np.linalg.norm(whole_ch) for s in sub_ch]
        flat = np.fft.fft(S.reconstruct(cf0, tr0, S.generate_channels(cf0, tr0)))
        rec = {"none": S.reconstruct(cfg, tr, ch), "whole": S.reconstruct(cfg, tr, whole_ch),
               "sub": S.reconstruct_subband(cfg, tr, ch, sata_osf=G.SATA_OSF)}
        fa = np.fft.fftfreq(cfg.Na, 1 / cfg.prf); P = cfg.PRF_op
        masks = [(fa >= -cfg.prf / 2 + k * P) & (fa < -cfg.prf / 2 + (k + 1) * P) & (np.abs(fa) <= cfg.abw / 2)
                 for k in range(cfg.Nrx)]
        esr = {k: [20 * np.log10(np.linalg.norm((np.fft.fft(v) - flat)[m]) / np.linalg.norm(flat[m])) for m in masks]
               for k, v in rec.items()}
        out.append((name, dch, esr))
    fig, A = plt.subplots(1, len(out), figsize=(17, 4.4), sharey=True, constrained_layout=True)
    lab = {"none": "no SATA", "whole": "SATA whole band", "sub": "SATA per sub-band"}
    for a, (name, dch, esr) in zip(A, out):
        for j, k in enumerate(("none", "whole", "sub")):
            a.bar(np.arange(4) + (j - 1) * 0.27, esr[k], 0.27, color=C[k], label=lab[k])
        a.set_xticks(range(4), [f"band {k}\n{(-1.5 + k) * 500:+.0f} Hz" for k in range(4)], fontsize=8)
        a.set_title(name + "\ncorrected channels, set k vs whole band:\n" + ", ".join(f"{d * 100:.2f}%" for d in dch),
                    fontsize=8.5)
        a.grid(alpha=.3, axis="y"); a.invert_yaxis()
    A[0].set_ylabel("error of the output sub-band\nvs flat reconstruction [dB]"); A[-1].legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(OUT, "band_errors.png"), dpi=120); plt.close(fig)
    return out


def fig_metric():
    fig, axs = plt.subplots(2, 1, figsize=(15, 8.4), constrained_layout=True)
    c = G.Case("ramp", "N5_a20", "", G.heights_from_slopes([20] * 4, 200.0), 2600.0, {})
    cfg, foc = V.simulate(c)
    c0 = S.make_config(list(zip(c.xs, c.heights)), bxt_max=0.0); t0 = S.build_platform_tracks(c0)
    sref = S.generate_reference(c0, t0, c0.scene.ptg[None, :]); H = np.conj(np.fft.fft(sref))
    ideal = np.roll(np.fft.ifft(np.fft.fft(S.generate_reference(c0, t0)) * H), cfg.Na // 2)
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf; pk = np.abs(foc["flat"]).max()
    d = lambda x: 20 * np.log10(np.abs(x) / pk + 1e-12)
    m, _, _ = V.metrics(cfg, foc, c)
    a = axs[0]
    a.plot(t, d(ideal), color="0.35", lw=1.0, label="ideal monostatic signal (previous reference, not band-limited)")
    a.plot(t, d(foc["flat"]), color=C["flat"], lw=1.2, label="flat-terrain reconstruction ($b_{xt}$ = 0; new reference)")
    a.plot(t, d(foc["flat"] - ideal), color="k", lw=0.7, ls=":", label="difference of the two")
    a.set_title("Case B-036 geometry ($N$ = 5, 2.6 km apart) WITHOUT topographic error: the reconstruction differs from the "
                "ideal signal by its band-edge skirts\n(processed band $B_a$ = 1235 Hz; ideal signal $\\approx$ 1340 Hz). "
                f"Read in the ambiguity windows (grey), this skirt gives {m['floor_db']:.1f} dB with no topography at all.",
                fontsize=9.5)
    b = axs[1]
    b.plot(t, d(foc["flat"]), color=C["flat"], lw=1.2, label="flat-terrain reconstruction")
    b.plot(t, d(foc["whole"]), color=C["whole"], lw=1.0, ls="--", label=f"SATA whole band: |s| (previous metric {m['ambpk_db_whole']:.1f} dB)")
    b.plot(t, d(foc["whole"] - foc["flat"]), color="k", lw=0.8,
           label=f"SATA whole band minus flat: topographic ambiguity ({m['ambt_db_whole']:.1f} dB)")
    b.set_title(f"Same case with heights $j\\cdot200\\tan20^\\circ$ and $b_{{xt}}$ = [63.7, 27.0, 4.1, 1.7] m: no SATA "
                f"{m['ambt_db_none']:.1f} dB, SATA whole band {m['ambt_db_whole']:.1f} dB, per sub-band {m['ambt_db_sub']:.1f} dB "
                "(topographic ambiguity)", fontsize=9.5)
    for a in axs:
        for x in c.xs:
            for mm in (-2, -1, 1, 2):
                a.axvspan(x / cfg.system.vs + mm * 0.8101 - 0.02, x / cfg.system.vs + mm * 0.8101 + 0.02, color="0.8", lw=0, zorder=0)
        a.set_xlim(-2.6, 2.6); a.set_ylim(-110, 5); a.grid(alpha=.3); a.legend(fontsize=8.5, loc="upper right")
        a.set_xlabel("Azimuth time [s]"); a.set_ylabel("[dB re. flat peak]")
    fig.savefig(os.path.join(OUT, "metric_v3.png"), dpi=115); plt.close(fig)


def fig_spacing():
    R = list(csv.DictReader(open(os.path.join(V.OUT, "spacing.csv"))))
    names = {"ramp_n20": "along-track ramp N = 5, 20 deg, normal heights (0-291 m)",
             "ramp_l20": "along-track ramp N = 5, 20 deg, large heights (0-3785 m)",
             "random500": "random heights N = 5, $h_{max}$ = 500 m (seed 0)"}
    fig, axs = plt.subplots(3, 1, figsize=(13, 11.5), constrained_layout=True)
    for a, (key, title) in zip(axs, names.items()):
        r = [x for x in R if x["config"] == key]
        s = np.array([float(x["spacing"]) for x in r]) / 1e3
        for k, lab, col, ls in (("ambt_db_none", "no SATA", C["none"], "--"), ("ambt_db_whole", "SATA whole band", C["whole"], "-"),
                                ("ambt_db_sub", "SATA per sub-band", C["sub"], ":"), ("floor_db", "floor of the flat reconstruction", "0.4", "-.")):
            a.plot(s, [float(x[k]) for x in r], ls, marker="o", ms=4, color=col, label=lab, lw=1.6)
        for x in r:
            sp = float(x["spacing"])
            if V.collides(sp, 5):
                a.axvspan(sp / 1e3 - 0.12, sp / 1e3 + 0.12, color="0.85", zorder=0)
        a.axvline(2.52, color="k", lw=0.7, ls=":")
        a.set_title(title, fontsize=10); a.grid(alpha=.3); a.set_ylim(-70, 0)
        a.set_ylabel("topographic ambiguity [dB]"); a.legend(fontsize=8, loc="lower right", ncol=2)
    axs[-1].set_xlabel("spacing between neighbouring targets [km]  (dotted: $2h_w$ = 2.52 km; grey: a pair of targets "
                       "within 0.63 km of a multiple of 6.23 km)")
    fig.savefig(os.path.join(OUT, "spacing_v3.png"), dpi=115); plt.close(fig)


def fig_saturation_single():
    R = list(csv.DictReader(open("probes/saturation_single.csv")))
    h = np.array([float(r["h"]) for r in R]) / 1e3
    g = lambda k: np.array([float(r[k]) for r in R])
    b1, b2 = 0.595, 0.190
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    a = axs[0]
    for i, (col, bx) in enumerate(zip(("#6A1B9A", "#1565C0", "#00897B", "#9E9D24"), (63.7, 27.0, 4.1, 1.7))):
        a.plot(h, np.degrees(g(f"phi{i}")), ".", ms=3, color=col, label=f"channel {i}, $b_{{xt}}$ = {bx} m")
    a.set_ylabel("phase error $\\varphi_i$ (wrapped) [deg]"); a.set_ylim(-185, 185); a.grid(alpha=.3)
    a.legend(fontsize=8, ncol=4, loc="lower left")
    a.set_title("One target at x = 0, height h, no SATA. Top: phase error of each channel "
                "($\\varphi_i = 2\\pi\\,\\Delta C_{0,i}/\\lambda$, one cycle every 1.1 / 2.6 / 17 / 42 km of height)", fontsize=10)
    a = axs[1]
    meas1 = np.maximum(g("meas_m-1"), g("meas_m1")); meas2 = np.maximum(g("meas_m-2"), g("meas_m2"))
    pred1 = 20 * np.log10(np.maximum(g("c1"), g("c3")) * b1 + 1e-12); pred2 = 20 * np.log10(g("c2") * b2 + 1e-12)
    a.plot(h, meas1, color=C["none"], lw=1.6, label="measured, windows $m = \\pm1$")
    a.plot(h, pred1, "k--", lw=1.0, label="predicted $\\max_{d=\\pm1}|c(d)|\\,\\beta_1$")
    a.plot(h, meas2, color="#F9A825", lw=1.4, label="measured, windows $m = \\pm2$")
    a.plot(h, pred2, "k:", lw=1.0, label="predicted $|c(2)|\\,\\beta_2$")
    a.plot(h, 20 * np.log10(g("c0")), color=C["flat"], lw=1.0, alpha=.7, label="main peak loss $|c(0)|$")
    a.axhline(20 * np.log10(b1), color="k", lw=1.2); a.text(5.95, 20 * np.log10(b1) + 0.8, "ceiling $20\\log\\beta_1$ = -4.5 dB", ha="right", fontsize=9)
    a.axhline(10 * np.log10(0.25) + 20 * np.log10(b1), color="0.4", lw=1.0, ls="-.")
    a.text(5.95, 10 * np.log10(0.25) + 20 * np.log10(b1) + 0.8, "random phases, rms: -10.5 dB", ha="right", fontsize=9, color="0.3")
    a.set_ylim(-45, 2); a.grid(alpha=.3); a.set_xlabel("target height h [km]")
    a.set_ylabel("[dB re. flat peak]"); a.legend(fontsize=8, ncol=3, loc="lower right")
    a.set_title("Bottom: ambiguity measured in the simulation (topographic ambiguity) and predicted from the four phases alone", fontsize=10)
    fig.savefig(os.path.join(OUT, "saturation_single.png"), dpi=115); plt.close(fig)


def montecarlo(n_trials=20000, hmax=20000.0, seed=0):
    h2pi = np.array([1.101, 2.599, 17.14, 42.35]) * 1e3
    rng = np.random.default_rng(seed)
    out = {}
    D = np.exp(-2j * np.pi * np.outer(np.arange(4), np.arange(4)) / 4)
    for N in (1, 2, 3, 5, 7, 9, 13):
        hs = rng.uniform(0, hmax, (n_trials, N))
        ph = np.exp(2j * np.pi * hs[..., None] / h2pi)
        c = np.abs(ph @ D) / 4
        lvl = np.maximum(np.maximum(c[..., 1], c[..., 3]) * 0.595, c[..., 2] * 0.190).max(axis=1)
        out[N] = 20 * np.log10(np.percentile(lvl, [10, 50, 90]))
    return out


def fig_sata_gain():
    R = list(csv.DictReader(open("probes/sata_gain.csv")))
    R = [r for r in R if r["step"] == "2600" and r["N"] == "5"]
    dh = np.array([float(r["dh_neighbour"]) for r in R])
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    for i, (col, bx) in enumerate(zip(("#6A1B9A", "#1565C0", "#00897B", "#9E9D24"), (63.7, 27.0, 4.1, 1.7))):
        axs[0].plot(dh, [float(r[f"absg_ch{i}"]) for r in R], "o-", color=col, label=f"channel {i}, $b_{{xt}}$ = {bx} m")
    axs[0].set_xscale("log"); axs[0].set_xlabel("height step between neighbours 200 m apart [m]")
    axs[0].set_ylabel("$|g_i|$ (mean over the targets)"); axs[0].grid(alpha=.3, which="both"); axs[0].legend(fontsize=8)
    axs[0].set_title("Effective gain of each channel after SATA (ideal: 1)", fontsize=10)
    axs[1].plot(dh, [float(r["meas_whole_db"]) for r in R], "o-", color=C["whole"], label="SATA whole band, measured")
    axs[1].plot(dh, [float(r["pred_db"]) for r in R], "k--", label="predicted from the gains $g_i$")
    axs[1].plot(dh, [float(r["meas_none_db"]) for r in R], "s-", color=C["none"], label="no SATA, measured")
    axs[1].axhline(20 * np.log10(np.sqrt(2) / 4 * 0.595), color="0.4", ls="-.", lw=1)
    axs[1].text(dh[0], 20 * np.log10(np.sqrt(2) / 4 * 0.595) + 0.8, "$g$ = (0, 0, 1, 1): -13.5 dB", fontsize=8.5, color="0.3")
    axs[1].set_xscale("log"); axs[1].set_xlabel("height step between neighbours 200 m apart [m]")
    axs[1].set_ylabel("topographic ambiguity [dB]"); axs[1].grid(alpha=.3, which="both"); axs[1].legend(fontsize=8)
    axs[1].set_title("Configuration C, N = 5 (large heights, 200 m apart)", fontsize=10)
    fig.savefig(os.path.join(OUT, "sata_gain.png"), dpi=115); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_tf(); print("tf")
    rows = table_lookangle(); print(rows)
    with open(os.path.join(OUT, "lookangle.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    be = band_errors(); print([(n, [round(x * 100, 2) for x in d]) for n, d, _ in be])
    fig_metric(); print("metric")
    fig_spacing(); print("spacing")
    fig_saturation_single(); print("sat")
    mc = montecarlo(); print({k: np.round(v, 1) for k, v in mc.items()})
    with open(os.path.join(OUT, "montecarlo.csv"), "w") as fh:
        fh.write("N,p10,p50,p90\n" + "".join(f"{k},{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}\n" for k, v in mc.items()))
    fig_sata_gain(); print("gain")
