"""
Labels and indexes of the whole-band and per-sub-band SATA kernels, measured end to end.

Part 1  which parcel of a target does each kernel read m folds away from the target?
        Single target on a flat scene (its own delta_C0 = 0); the table is zero except a block of
        constant value at x_j + m X. The ambiguity that appears measures the energy fraction of
        the target that the kernel reads at that block.
Part 2  leakage against the mis-phased fraction of one parcel: block at x_j + X of growing
        half-width under the whole-band kernel.
Part 3  per-output-band leakage model (inversion filters + per-channel phase errors) against the
        per-band ambiguity measured on a real terrain case.

    python probes/probe_subband_labels.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S
import sata1d.sata as SS
from sata1d.subband import sata_1d_subband, subband_frequency_beam, subband_axis, sata_channels_subband, channel_spectrum
from sata1d.reconstruction import reconstruct_from_spectra, filter_matrix, GetInversionFilters
from run_one_case import ambiguity_level
from run_geometry_sweep import SATA_OSF, BXT_MAX, BXT_SEED
from run_terrain_lut_cases import terrain_from, target_positions, ALL_PRESETS, X_FOLD

OUT = os.path.join(os.path.dirname(__file__), "subband_labels")
os.makedirs(OUT, exist_ok=True)


def setup(targets, terrain=None):
    hs = [0.0] * len(targets) if terrain is None else list(terrain(np.asarray(targets, float)))
    cfg = S.make_config(list(zip(targets, hs)), bxt_max=BXT_MAX, seed=BXT_SEED)
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    Na = cfg.Na
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), Na // 2))
    t = (np.arange(Na) - Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    pk = F(S.generate_reference(cfg, tr)).max()
    xs = np.asarray(targets, float)
    amb = lambda s: ambiguity_level(t, F(s), pk, xs, cfg.system.vs, dt)
    return cfg, tr, ch, F, amb, pk, t


def whole(cfg, ch, M):
    out = ch.copy()
    for i in range(cfg.Nrx):
        out[i] = SS.sata_1d(out[i], M[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                            r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=SATA_OSF)
    return out


def sub_sets(cfg, tr, ch, M):
    return [sata_channels_subband(cfg, tr, ch, k, sata_osf=SATA_OSF, maps=list(M)) for k in range(cfg.Nsb)]


def recon(cfg, tr, sets):
    spec = [[channel_spectrum(cfg, sk[j]) for j in range(cfg.Nrx)] for sk in sets]
    return reconstruct_from_spectra(cfg, tr, spec, True)


def cell_of(cfg, x, i):
    return int(np.clip(SS.az_pixel_of_scatterer(cfg, x, i), 0, cfg.Na_ch - 1))


def block_tables(cfg, x_blocks, value, halfwidth):
    """One table per channel: value * b_xt,i / b_xt,0 [m] (a real delta_C0 scales with b_xt) on the cells
    |c - cell(x_b)| <= halfwidth, zero elsewhere."""
    M = []
    for i in range(cfg.Nrx):
        arr = np.zeros(cfg.Na_ch)
        for xb in x_blocks:
            c = cell_of(cfg, xb, i)
            arr[max(0, c - halfwidth):min(cfg.Na_ch, c + halfwidth + 1)] = value * cfg.array.bxt[i] / cfg.array.bxt[0]
        M.append(arr)
    return M


def band_decomposition(cfg, rec, amb):
    U = np.fft.fft(rec); fa = np.fft.fftfreq(cfg.Na, d=1.0 / cfg.prf); vals = []
    for k in range(cfg.Nsb):
        f_k = (-cfg.Nsb / 2 + k + 0.5) * cfg.PRF_op
        m = (fa >= f_k - cfg.PRF_op / 2) & (fa < f_k + cfg.PRF_op / 2)
        vals.append(amb(np.fft.ifft(U * m)))
    return vals


# ----------------------------------------------------------------------------------------------
def part0_axes(cfg):
    """The labels of the two axis functions: for a true Doppler f, where (in folds X) is the table read?"""
    prf, v, wl, r = cfg.PRF_op, cfg.system.vs, cfg.system.wl, cfg.scene.r0
    Tsub, hop, Nzp = SS.subaperture_sizing(r, prf, v, wl, SATA_OSF)
    a_cells = lambda f: r * np.tan(np.arcsin(wl * f / (2 * v))) / v * prf
    f = np.linspace(-640, 640, 1281)
    phi = (f + prf / 2) % prf - prf / 2                       # baseband of the channel
    b = np.round(phi * Nzp / prf).astype(int) % Nzp           # FFT bin (fft order)
    rows = {}
    X_cells = X_FOLD / (v / prf)
    fs_w, az_w = SS.subaperture_axis(prf, Nzp, v, wl, r, cfg.sq_tx)
    rows["whole"] = (az_w[b] - a_cells(f)) / X_cells
    for k in range(cfg.Nsb):
        f_k = subband_frequency_beam(cfg, k)[0]
        fs_k, az_k = subband_axis(prf, Nzp, v, wl, r, f_k)
        rows[f"sub k={k}"] = (az_k[b] - a_cells(f)) / X_cells
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for (name, off), ls in zip(rows.items(), ["-", "--", "--", "--", "--"]):
        ax.plot(f, off, ls=ls, lw=1.8, label=name)
    for e in (-500, -250, 0, 250, 500):
        ax.axvline(e, color="0.8", lw=0.7)
    ax.set_xlabel("true Doppler of the energy [Hz]"); ax.set_ylabel("table read at  x_true + (offset) X")
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3]); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=5, loc="upper center")
    ax.set_title("Where each kernel reads the table, from the axis functions (subaperture_axis / subband_axis)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "part0_axes.png"), dpi=110); plt.close(fig)
    print("Part 0 (axis functions): read offset in folds X, by true-Doppler band")
    bands = [(-640, -500), (-500, -250), (-250, 0), (0, 250), (250, 500), (500, 640)]
    print(f"  {'kernel':10s}" + "".join(f"{f'[{lo},{hi})':>14s}" for lo, hi in bands))
    for name, off in rows.items():
        cells = []
        for lo, hi in bands:
            sel = (f >= lo) & (f < hi)
            o = off[sel]; cells.append(f"{np.round(np.median(o)):+.0f} (sd {np.std(o):.2f})")
        print(f"  {name:10s}" + "".join(f"{c:>14s}" for c in cells))


def part1_blocks():
    print("\nPart 1 (end to end): single target at x = 0, flat scene, block of 30 mm at x + m X, half-width 150 cells")
    cfg, tr, ch, F, amb, pk, t = setup([0.0])
    print(f"  no table: {amb(S.reconstruct(cfg, tr, ch)):.1f} dB")
    hdr = f"  {'kernel':10s}" + "".join(f"{f'm={m:+d}':>9s}" for m in range(-3, 4))
    print(hdr)
    res = {}
    for name in ["whole"] + [f"sub k={k}" for k in range(cfg.Nsb)]:
        vals = []
        for m in range(-3, 4):
            M = block_tables(cfg, [m * X_FOLD], 0.030, 150)
            if name == "whole":
                rec = S.reconstruct(cfg, tr, whole(cfg, ch, M))
            else:
                k = int(name[-1])
                sets = [ch] * cfg.Nsb
                sets[k] = sata_channels_subband(cfg, tr, ch, k, sata_osf=SATA_OSF, maps=M)
                rec = recon(cfg, tr, sets)
            vals.append(amb(rec))
        res[name] = vals
        print(f"  {name:10s}" + "".join(f"{v:9.1f}" for v in vals))
    print("  (an ambiguity appears only where the kernel reads part of the target's energy; its level grows with that energy share)")
    return res


def part2_fraction():
    print("\nPart 2: whole-band kernel, block of 30 mm at x + X with growing half-width (cells); 1 cell = 1.23 Hz of Doppler")
    cfg, tr, ch, F, amb, pk, t = setup([0.0])
    print(f"  {'half-width':>10s} {'Doppler span [Hz]':>18s} {'ambiguity [dB]':>15s}")
    out = []
    for w in (10, 20, 40, 80, 160, 320, 640):
        M = block_tables(cfg, [X_FOLD], 0.030, w)
        a = amb(S.reconstruct(cfg, tr, whole(cfg, ch, M)))
        out.append((w, a))
        print(f"  {w:10d} {2 * w * cfg.PRF_op / X_FOLD:18.0f} {a:15.1f}")
    return out


def part3_model(cid, n_targets, centre=-100.0):
    expr, desc = ALL_PRESETS[cid]
    terrain = terrain_from(expr)
    xs = target_positions(n_targets, centre=centre)
    cfg, tr, ch, F, amb, pk, t = setup(list(xs), terrain)
    tab = [S.terrain_lookup_table(cfg, tr, i, terrain, step=4) for i in range(cfg.Nrx)]
    dC0 = S.delta_C0_per_target(cfg, tr)
    rec_w = S.reconstruct(cfg, tr, whole(cfg, ch, tab))
    rec_s = recon(cfg, tr, sub_sets(cfg, tr, ch, tab))
    meas_w = band_decomposition(cfg, rec_w, amb); meas_s = band_decomposition(cfg, rec_s, amb)
    # phase error of parcel q under kernel k (read at x + (k - q) X), channel i, target j
    wl = cfg.system.wl
    err = np.zeros((cfg.Nsb, cfg.Nsb, cfg.Nrx, len(xs)))            # [k, q, i, j] in metres
    for k in range(cfg.Nsb):
        for q in range(cfg.Nsb):
            for i in range(cfg.Nrx):
                for j, x in enumerate(xs):
                    c = cell_of(cfg, x + (k - q) * X_FOLD, i)
                    err[k, q, i, j] = tab[i][c] - dC0[j, i]
    # whole-band kernel: parcel q is read with offset m(f) = 0 for |f| < 250, +1 for f < -250, -1 for f > 250
    # -> energy fractions of each parcel that are right / one fold off (Doppler band +-640 Hz, flat spectrum)
    B = 640.0
    edges = [(-1000, -500), (-500, 0), (0, 500), (500, 1000)]
    frac = []                        # per parcel q: (fraction read at m=0, m=+1, m=-1) under the whole kernel
    for lo, hi in edges:
        lo, hi = max(lo, -B), min(hi, B)
        f0 = max(0.0, min(hi, 250) - max(lo, -250)); fp = max(0.0, min(hi, -250) - lo); fm = max(0.0, hi - max(lo, 250))
        frac.append((f0 / (2 * B), fp / (2 * B), fm / (2 * B)))
    share = [(min(hi, B) - max(lo, -B)) / (2 * B) for lo, hi in edges]
    # leakage coefficients from the inversion at the band centres
    prfCh, Nsb = cfg.PRF_op, cfg.Nsb
    fsub = np.array([(-Nsb / 2 + 0.5) * prfCh])                  # one frequency: centre of band 0 -> all bands via +q PRF_op
    Hf = filter_matrix(cfg, tr, fsub)                             # [Nf, Nsb(out band), Nrx]?  use the API as in reconstruct
    iHf = GetInversionFilters(Hf)                                  # [Nsb*Nf, Nrx]
    Hm = Hf[0]                                                     # [Nsb, Nrx] : H_i(f_q) with q = row
    G = iHf.reshape(Nsb, -1, cfg.Nrx)[:, 0, :]                    # [k, i]
    def leak(k, q, m_err):                                        # |sum_i G_k,i H_i(f_q) (e^{j psi_i} - 1)|, psi = 2 pi err / wl
        return abs(np.sum(G[k] * Hm[q] * (np.exp(2j * np.pi * m_err / wl) - 1.0)))
    print(f"\nPart 3: {cid} ({desc}), {n_targets} targets; per-channel table error of parcel q under kernel k [mm], mean over targets:")
    for k in range(Nsb):
        print(f"  kernel k={k}: " + "  ".join(f"q={q}: [" + ",".join(f"{1e3 * err[k, q, i, :].mean():6.1f}" for i in range(cfg.Nrx)) + "]" for q in range(Nsb)))
    print("  per output band k: leakage of parcel q lands at ambiguity order q - k; the metric takes the largest (orders +-1, +-2).")
    print("  predicted = fraction of parcel q that is mis-phased x |sum_i G_k,i H_i(f_q) (e^{j psi_q,i} - 1)| / Nrx, mean over targets")
    print(f"  {'band k':>7s} {'pred whole':>11s} {'(from q)':>8s} {'pred sub':>9s} {'(from q)':>8s} {'ratio dB':>9s} | {'meas whole':>11s} {'meas sub':>9s} {'diff dB':>8s}")
    for k in range(Nsb):
        Lw, Ls = {}, {}
        for q in range(Nsb):
            if q == k or abs(q - k) > 2:
                continue
            lw = ls = 0.0
            for j in range(len(xs)):
                e_p = np.array([tab[i][cell_of(cfg, xs[j] + X_FOLD, i)] - dC0[j, i] for i in range(cfg.Nrx)])
                e_m = np.array([tab[i][cell_of(cfg, xs[j] - X_FOLD, i)] - dC0[j, i] for i in range(cfg.Nrx)])
                lw += frac[q][1] * leak(k, q, e_p) + frac[q][2] * leak(k, q, e_m)
                ls += share[q] * leak(k, q, err[k, q, :, j])
            Lw[q] = lw / (len(xs) * cfg.Nrx); Ls[q] = ls / (len(xs) * cfg.Nrx)
        qw = max(Lw, key=Lw.get); qs = max(Ls, key=Ls.get)
        print(f"  {k:7d} {20 * np.log10(Lw[qw] + 1e-12):11.1f} {qw:8d} {20 * np.log10(Ls[qs] + 1e-12):9.1f} {qs:8d} "
              f"{20 * np.log10((Ls[qs] + 1e-12) / (Lw[qw] + 1e-12)):9.1f} | {meas_w[k]:11.1f} {meas_s[k]:9.1f} {meas_s[k] - meas_w[k]:8.1f}")
    print(f"  total: whole {amb(rec_w):.1f} dB, sub {amb(rec_s):.1f} dB, no SATA {amb(S.reconstruct(cfg, tr, ch)):.1f} dB")


if __name__ == "__main__":
    t0 = time.time()
    cfg0 = S.make_config([(0.0, 0.0)], bxt_max=BXT_MAX, seed=BXT_SEED)
    part0_axes(cfg0)
    part1_blocks()
    part2_fraction()
    part3_model("SIN-B50-A50-P30", 1)
    part3_model("SIN-B50-A50-P30", 7)
    part3_model("RAMP-H1000-L12", 7)
    print(f"\n({time.time() - t0:.0f} s)")
