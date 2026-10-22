"""
The joint (occupied parcels and full system) for other numbers of channels: Nrx = 2, 3, 4, 5, 6, 8 at the same total
PRF (2000 Hz, so PRF_op = 2000 / Nrx and the DPCA along-track spacing stays d = v / PRF). Channel 0 on the
transmitter; cross-track baselines bxt ~ U(0, bmax) with the sweep seed (the first channels are the same for every
Nrx). Terrain cases use the DEM table, ramps the nearest-target map, as in the sweep.
    SATA_LUT_CACHE=/tmp/lutcache python joint_cases/nrx_cases.py --budget 140      # from sata1d_code, repeat
Pages: joint_cases/nrx/<bmax>/<case>_N<Nrx>.png (+ .json); summary and PDF: joint_cases/nrx/nrx_cases.pdf
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
from sata1d.config import ArrayGeometry
import run_geometry_sweep as G
import run_joint_sweep as W
import joint_sweep_summary as JS
import sweep_5panel as P5

OUT = os.path.join(HERE, "nrx")
CASES = ["C05", "C15", "C24", "C31"]
NRX = [2, 3, 4, 5, 6, 8]
BMAX = [100.0, 500.0]
CACHE = os.environ.get("SATA_LUT_CACHE", "/tmp/lutcache")
VERSION = 2          # 2: reconstruct_joint_any_nrx (parcel grid that also works for an odd number of channels)


def reconstruct_joint_any_nrx(cfg, tracks, s_channel, maps=None, dC0=None, sata_osf=4, zeroOutBw=True, occupied=True):
    """sata1d.joint.reconstruct_joint with one change: the parcel frequencies are f_q(b) = -Nsb P / 2 + ((phi_b +
    Nsb P / 2) mod P) + q P, which are congruent to the bin frequency phi_b = b P / Nzp for any Nsb (the original
    -Nsb P / 2 + q P + phi_b is off by P / 2 when Nsb is odd), and the parcels are put back at the FFT index of their own
    frequency. For an even Nsb both are identical to the original."""
    from sata1d.joint import reference_coefficients, joint_maps
    from sata1d.sata import subaperture_sizing, triangular_window
    Nrx, Nsb, Na_ch, P = cfg.Nrx, cfg.Nsb, cfg.Na_ch, cfg.PRF_op
    wl, v, r0 = cfg.system.wl, cfg.system.vs, cfg.scene.r0
    Na = Na_ch * Nsb
    s = np.asarray(s_channel, dtype=complex)
    maps = np.asarray(joint_maps(cfg, tracks, dC0) if maps is None else maps, float)
    coef = reference_coefficients(cfg, tracks)
    Tsub, hop, Nzp = subaperture_sizing(r0, P, v, wl, sata_osf)
    win = triangular_window(Tsub)
    phi = np.arange(Nzp) * P / Nzp
    f = -Nsb * P / 2.0 + np.mod(phi + Nsb * P / 2.0, P)[None, :] + (np.arange(Nsb) * P)[:, None]     # [Nsb, Nzp]
    a = r0 * np.tan(np.arcsin(np.clip(wl * f / (2.0 * v), -1.0, 1.0))) / v * P
    Hq = np.zeros((Nzp, Nrx, Nsb), complex)
    for i in range(Nrx):
        C0, C1, C2, Dt = coef[i]
        Hq[:, i, :] = np.exp(-2j * np.pi * (C0 / wl + (-C1 + Dt) * f.T + C2 * wl * f.T ** 2))
    if occupied:
        B = 4.0 * v * np.sin(cfg.theta_tx / 2.0) / wl
        occ = np.abs(f.T) <= B / 2.0 + 4.0 * P / Tsub
    lo = (Nzp - Tsub) // 2
    L, Lb = Tsub * Nsb, Nzp * Nsb
    idx = np.mod(np.round(f / (P / Nzp)).astype(int), Lb).reshape(-1)        # FFT index of every (parcel, bin)
    wfull = np.interp(np.arange(L) / Nsb, np.arange(Tsub), win)
    pad = lo * Nsb
    out = np.zeros(Na + Lb + 2 * pad, complex)
    wsum = np.zeros(Na + Lb + 2 * pad)
    spad = np.concatenate([s, np.zeros((Nrx, Tsub), complex)], axis=1)
    buf = np.zeros((Nrx, Nzp), complex)
    U = np.zeros(Lb, complex)
    for start in range(0, Na_ch, hop):
        c = start + 0.5 * Tsub
        buf[:] = 0.0
        buf[:, lo:lo + Tsub] = spad[:, start:start + Tsub] * win
        S_ = np.fft.fft(buf, axis=1)
        pos = np.clip(np.round(c + a).astype(int), 0, Na_ch - 1)
        G = Hq * np.exp(-2j * np.pi * maps[:, pos].transpose(2, 0, 1) / wl)
        if occupied:
            u = Nsb * (np.linalg.pinv(G * occ[:, None, :]) @ S_.T[:, :, None])[:, :, 0]
        else:
            u = Nsb * np.linalg.solve(G, S_.T[:, :, None])[:, :, 0]
        U[:] = 0.0
        U[idx] = u.T.reshape(-1)
        y = np.fft.ifft(U)
        i0 = (start - lo) * Nsb + pad
        out[i0:i0 + Lb] += y
        wsum[pad + start * Nsb:pad + start * Nsb + L] += wfull
    out = out[pad:pad + Na]; wsum = wsum[pad:pad + Na]
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    if zeroOutBw:
        fa = np.fft.fftfreq(Na, d=1.0 / cfg.prf)
        Uo = np.fft.fft(out); Uo[np.abs(fa) > cfg.abw / 2] = 0
        out = np.fft.ifft(Uo)
    return out


P5.reconstruct_joint = reconstruct_joint_any_nrx


def lut(cfg, tr, i, terrain, cid):
    """Terrain table cached per case, channel, baselines and line length (the length changes with Nrx)."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"nrx_{cid}_ch{i}_bat{cfg.array.bat[i]:.3f}_bxt{cfg.array.bxt[i]:g}_n{cfg.Na_ch}.npy")
    if os.path.exists(f):
        return np.load(f)
    m = S.terrain_lookup_table(cfg, tr, i, terrain, step=4)
    np.save(f, m)
    return m


def config(cid, nrx, bmax):
    c = W.case_info(cid)
    cfg = S.make_config(list(zip(c["xs"], c["hs"])), nrx=nrx, bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    b = np.random.default_rng(G.BXT_SEED).uniform(0.0, bmax, nrx).round(1)
    b[0] = 0.0
    return replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=b))


def summary(out):
    res = {}
    for bm in BMAX:
        for cid in CASES:
            for n in NRX:
                f = os.path.join(out, f"b{bm:g}", f"{cid}_N{n}.json")
                if os.path.exists(f):
                    res[(bm, cid, n)] = json.load(open(f))
    fig, axs = plt.subplots(len(BMAX), len(CASES), figsize=(22, 9.5), sharex=True, sharey=True)
    for r, bm in enumerate(BMAX):
        for k, cid in enumerate(CASES):
            ax = axs[r, k]
            ns = [n for n in NRX if (bm, cid, n) in res]
            if not ns:
                continue
            g = lambda key: [res[(bm, cid, n)][key] for n in ns]
            ax.plot(ns, g("ref"), "o-", color="tab:green", lw=2, label="Reference")
            ax.plot(ns, g("none"), "s--", color="tab:red", label="no SATA")
            ax.plot(ns, g("whole"), "^:", color="tab:blue", label="SATA whole band (block map)")
            ax.plot(ns, g("joint_full"), "d--", color="tab:pink", label="joint, full system")
            ax.plot(ns, g("joint_occ"), "o-", color="tab:purple", lw=2, label="joint, occupied parcels")
            ax.set_xticks(NRX); ax.grid(alpha=0.3); ax.set_ylim(-75, 25)
            ax.set_title(f"{res[(bm, cid, ns[0])]['label'][:55]}\n$b_{{xt}}$ ~ U(0, {bm:g} m)", fontsize=9.5)
            if r == len(BMAX) - 1:
                ax.set_xlabel("number of channels $N_{rx}$ (PRF = 2000 Hz, PRF$_{op}$ = 2000 / $N_{rx}$)")
            if k == 0:
                ax.set_ylabel("highest ambiguity [dB]")
    axs[0, 0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Joint reconstruction against the number of channels (same total PRF, DPCA, channel 0 on the transmitter)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out, "summary.png"); fig.savefig(p, dpi=85); plt.close(fig)
    pngs, outline = [p], [("Summary", 0, 0)]
    for bm in BMAX:
        for cid in CASES:
            for n in NRX:
                if (bm, cid, n) in res:
                    outline.append((f"U(0, {bm:g} m) | {res[(bm, cid, n)]['label'][:50]} | Nrx = {n}", len(pngs), 0))
                    pngs.append(os.path.join(out, f"b{bm:g}", f"{cid}_N{n}.png"))
    return JS.bundle(pngs, os.path.join(out, "nrx_cases.pdf"), outline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return print("wrote", summary(OUT))
    deadline = time.time() + a.budget
    for bm in BMAX:
        for cid in CASES:
            for n in NRX:
                d = os.path.join(OUT, f"b{bm:g}")
                fj = os.path.join(d, f"{cid}_N{n}.json")
                if os.path.exists(fj) and json.load(open(fj)).get("version") == VERSION:
                    continue
                if time.time() > deadline:
                    return print("budget reached, run again", flush=True)
                cfg = config(cid, n, bm)
                tag = f"b{bm:g}"
                p, amb = P5.run(cid, tag, out=OUT, cfg=cfg, kind=f"random, U(0, {bm:g} m), Nrx = {n}", lut_fn=lut)
                os.replace(p, os.path.join(d, f"{cid}_N{n}.png"))
                json.dump(dict(amb, case=cid, nrx=n, bmax=bm, version=VERSION, bxt=cfg.array.bxt.tolist(), label=W.case_info(cid)["label"]),
                          open(fj, "w"), indent=1)
    print("all done:", summary(OUT), flush=True)


if __name__ == "__main__":
    main()
