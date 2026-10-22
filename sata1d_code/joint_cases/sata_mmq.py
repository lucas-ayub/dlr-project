"""
Test (outside the pipeline; no pipeline file is changed): SATA whole band and SATA per sub-band with the 4x4 inversion of
the standard reconstruction replaced by the solution used by the joint: regularised least squares (Tikhonov, energy of
the focused line), mu = 0.03 sigma_max, two passes (iterated Tikhonov).

Model of one bin phi of the channels (the joint's model with a flat DEM, block diagonal in frequency):
    S_i(phi) = (1/Nrx) sum_q H_i(f_q) K(f_q) G(f_q)         A_phi[i, q] = H_i(f_q) K(f_q) / Nrx
pass 1:  R1 = (A^H A + mu^2 I)^-1 A^H        two passes:  R2 = R1 + R1 (I - A R1)
output:  U(f_q) = K(f_q) [R2 S(phi)]_q, cut to B_a (as the joint: back to the raw line, U = K G).
mu = 0.03 sigma_max with sigma_max the same 12-iteration power-method estimate that the joint uses (flat DEM), so that
with no SATA this is exactly reconstruct_joint_line with a zero DEM (checked: --check).
SATA whole band: the SATA-corrected channels go through these filters. SATA per sub-band: output sub-band q takes its
row from the channels corrected for sub-band q (as reconstruct_from_spectra does with the inverse).

    python joint_cases/sata_mmq.py --check              # equivalence with the joint (flat DEM), one case
    python joint_cases/sata_mmq.py --workers 4          # the 86 cases of sweep_final: numbers, table and pages
Output: joint_cases/sata_mmq/numbers/*.json, pages/*.png, sata_mmq.pdf, tabela.txt
"""
import argparse, json, os, sys, time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import sata1d as S
import sata1d.sata as SS
from sata1d.config import ArrayGeometry
from sata1d.joint_line import height_table, channel_maps, line_maps, LineModel, reconstruct_joint_line
from sata1d.subband import sata_channels_subband, channel_spectrum
from sata1d.reconstruction import filter_matrix, reconstruct_from_spectra
import audit_dem as AD
import sweep_final as SF
import run_joint_sweep as W

OUT = os.path.join(HERE, "sata_mmq")
PAGES = [("one_refpoint", "u500"), ("one_ramp8", "N4b300"), ("one_flank", "N8b100"), ("few_ramp15_9", "N4b500"),
         ("few_sin50km_5", "s50"), ("few_hills_5", "u100"), ("few_plateau_9", "s50"), ("few_ramp20_7", "N2b500"),
         ("many_mountains_25", "u100"), ("many_hills_49", "N8b100")]


def mmq_filters(cfg, tr, reg=0.03, passes=2):
    """W[b, q, i]: U(f_q) at bin b = sum_i W[b, q, i] S_i(b). Also returns (sigma used, exact largest sigma) in the
    joint's units (time domain)."""
    Na, L, N = cfg.Na, cfg.Na_ch, cfg.Nrx
    fsub = -cfg.prf / 2 + np.arange(L) * cfg.PRF_op / L
    hf = filter_matrix(cfg, tr, fsub).astype(complex)                     # [b, q, i] = H_i(f_q)
    K = np.fft.fft(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]))
    idx = (-(Na // 2) + np.arange(L)[:, None] + np.arange(N)[None, :] * L) % Na
    Kbq = K[idx]                                                           # [b, q]
    A = np.transpose(hf, (0, 2, 1)) * Kbq[:, None, :] / N                  # [b, i, q]
    s_exact = np.linalg.svd(A, compute_uv=False).max() * np.sqrt(N)        # per-bin units -> the joint's (time) units
    s_used = LineModel(cfg, tr, np.zeros((N, Na))).largest_singular_value()  # what the joint uses (flat DEM)
    mu2 = (reg * s_used / np.sqrt(N)) ** 2
    AH = np.conj(np.transpose(A, (0, 2, 1)))
    R1 = np.linalg.solve(AH @ A + mu2 * np.eye(N)[None], AH)              # [b, q, i]
    R = R1.copy()
    for _ in range(passes - 1):
        R = R + R1 @ (np.eye(N)[None] - A @ R)
    return Kbq[:, :, None] * R, s_used, s_exact


def from_spectra(cfg, Wf, spec):
    """As reconstruct_from_spectra, with the filters Wf instead of 4 H^-1. spec[q][i]: channel i for sub-band q."""
    Na, L, N = cfg.Na, cfg.Na_ch, cfg.Nrx
    srec = np.zeros(Na, complex)
    for q in range(cfg.Nsb):
        for i in range(N):
            srec[q * L:(q + 1) * L] += Wf[:, q, i] * spec[q][i]
    fa = -cfg.prf / 2 + np.arange(Na) * cfg.prf / Na
    srec[(fa < -cfg.abw / 2) | (fa > cfg.abw / 2)] = 0
    return np.fft.ifft(np.roll(srec, Na // 2))


def setup(sc, tag):
    """The same scene, channels and DEM tables as audit_dem.compute."""
    grp, fn = AD.SCENES[sc]
    c = fn()
    xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
    bxt, kind = AD.baseline(tag)
    cfg = S.make_config(list(zip(xs, hs)), nrx=len(bxt), bxt_max=0.0, amplitudes=c["amps"], tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hgrid = terrain(np.linspace(-half - 100.0, half + 100.0, 8001))
    tab = height_table(cfg, tr, hgrid.min(), hgrid.max())
    return c, cfg, tr, ch, channel_maps(cfg, tr, terrain, tab), line_maps(cfg, tr, terrain, tab), kind


def compute(sc, tag):
    c, cfg, tr, ch, Mch, Mline, kind = setup(sc, tag)
    Wf, s_used, s_exact = mmq_filters(cfg, tr)
    N = cfg.Nrx
    # SATA whole band: the corrected channels (same call as audit_dem.sata_whole)
    o = np.array(ch, complex).copy()
    for i in range(N):
        o[i] = SS.sata_1d(o[i], Mch[i], rref=cfg.scene.r0, prf=cfg.PRF_op, v=cfg.system.vs, wl=cfg.system.wl,
                          r=cfg.scene.r0, squint=cfg.sq_tx, inverse=True, sata_osf=AD.SATA_OSF)
    sp_none = [channel_spectrum(cfg, ch[i]) for i in range(N)]
    sp_whole = [channel_spectrum(cfg, o[i]) for i in range(N)]
    sp_sub = []
    for k in range(cfg.Nsb):                                               # same call as audit_dem.sata_sub
        corr = sata_channels_subband(cfg, tr, ch, k, sata_osf=AD.SATA_OSF, maps=list(Mch))
        sp_sub.append([channel_spectrum(cfg, corr[j]) for j in range(N)])
    raw = {"ref": S.generate_reference(cfg, tr),
           "none": S.reconstruct(cfg, tr, ch), "none_mmq": from_spectra(cfg, Wf, [sp_none] * cfg.Nsb),
           "whole": S.reconstruct(cfg, tr, o), "whole_mmq": from_spectra(cfg, Wf, [sp_whole] * cfg.Nsb),
           "sub": reconstruct_from_spectra(cfg, tr, sp_sub, True), "sub_mmq": from_spectra(cfg, Wf, sp_sub),
           "joint_line": reconstruct_joint_line(cfg, tr, ch, Mline)}
    Hf = np.conj(np.fft.fft(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])))
    env = {k: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2)) for k, s in raw.items()}
    pk = env["ref"].max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt_amb = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    orders = SF.orders_of(cfg)
    amb = {k: float(AD.ambiguity_level(t, v, pk, c["xs"], cfg.system.vs, dt_amb, orders=orders)) for k, v in env.items()}
    res = dict(amb, scene=sc, baseline=tag, nrx=N, bxt=[float(b) for b in cfg.array.bxt], label=c["label"],
               n_targets=int(len(c["xs"])), sigma_used=float(s_used), sigma_exact=float(s_exact))
    return res, dict(c=c, cfg=cfg, env=env, t=t, pk=pk, kind=kind, amb=amb)


STY = {"ref": ("Reference", dict(color="tab:green", lw=2.0)),
       "whole": ("SATA whole band (DEM), inversion", dict(color="tab:blue", lw=1.3, ls=(0, (5, 3)))),
       "whole_mmq": ("SATA whole band (DEM), regularised LS as the joint", dict(color="tab:cyan", lw=1.0)),
       "sub": ("SATA per sub-band (DEM), inversion", dict(color="tab:orange", lw=1.3, ls=(0, (5, 3)))),
       "sub_mmq": ("SATA per sub-band (DEM), regularised LS as the joint", dict(color="tab:brown", lw=1.0)),
       "joint_line": ("joint, final version", dict(color="tab:purple", lw=1.0))}


def page(res, d, out_png):
    cfg, env, t, pk, amb = d["cfg"], d["env"], d["t"], d["pk"], d["amb"]
    db = {k: 20 * np.log10(v / pk + 1e-12) for k, v in env.items()}
    fig, axs = plt.subplots(3, 1, figsize=(18, 13), sharex=True, gridspec_kw=dict(hspace=0.3, left=0.05, right=0.985, top=0.91, bottom=0.07))
    lab = res["label"] if res["label"].startswith("h =") else res["label"][0].upper() + res["label"][1:]
    fig.suptitle(f"{lab} | $N_{{rx}}$ = {cfg.Nrx}, $b_{{xt}}$ = {W.bxt_text(cfg.array.bxt)} ({d['kind']})\n"
                 "SATA followed by the inversion of the standard reconstruction vs. by regularised least squares as in the joint "
                 "($\\mu$ = 0.03 $\\sigma_{max}$, two passes)", fontsize=12.5)
    for ax, keys in zip(axs, (("ref", "whole", "whole_mmq"), ("ref", "sub", "sub_mmq"), ("ref", "joint_line"))):
        for k in keys:
            name, st = STY[k]
            ax.plot(t, db[k], label=name, **st)
        ax.set_title(" | ".join(f"{STY[k][0]} {amb[k]:.1f} dB" for k in keys), fontsize=10.5)
        ax.set_ylim(-90, 5); ax.set_xlim(t[0], t[-1]); ax.grid(alpha=0.3); ax.set_ylabel("[dB]"); ax.legend(fontsize=9, loc="upper left")
    axs[-1].set_xlabel("Azimuth time [s]")
    fig.text(0.5, 0.012, f"Highest ambiguity in dB relative to the reference peak (same measure as the sweep). No SATA: inversion {amb['none']:.1f} dB, "
             f"regularised LS {amb['none_mmq']:.1f} dB. Every method reads only the DEM.", ha="center", fontsize=9, color="0.35")
    fig.savefig(out_png, dpi=80); plt.close(fig)


def _job(case):
    sc, tag = case
    pj = os.path.join(OUT, "numbers", f"{sc}__{tag}.json")
    pp = os.path.join(OUT, "pages", f"{sc}__{tag}.png")
    if os.path.exists(pj) and ((sc, tag) not in PAGES or os.path.exists(pp)):
        return json.load(open(pj))
    res, d = compute(sc, tag)
    json.dump(res, open(pj, "w"), indent=1)
    if (sc, tag) in PAGES:
        page(res, d, os.path.join(OUT, "pages", f"{sc}__{tag}.png"))
    return res


def check():
    """With no SATA, these filters must give reconstruct_joint_line with a zero DEM."""
    c, cfg, tr, ch, Mch, Mline, kind = setup("few_hills_5", "u100")
    Wf, s_used, s_exact = mmq_filters(cfg, tr)
    a = from_spectra(cfg, Wf, [[channel_spectrum(cfg, ch[i]) for i in range(cfg.Nrx)]] * cfg.Nsb)
    b = reconstruct_joint_line(cfg, tr, ch, np.zeros((cfg.Nrx, cfg.Na)), iterations=400, tol=1e-12)
    s = S.reconstruct(cfg, tr, ch)
    print(f"sigma: used (12-it power method) {s_used:.3f}, exact {s_exact:.3f}")
    print(f"|filters - joint(flat DEM)| / |joint| = {np.linalg.norm(a - b) / np.linalg.norm(b):.2e}")
    print(f"|filters - standard inversion| / |standard| = {np.linalg.norm(a - s) / np.linalg.norm(s):.2e}")


def table(rows):
    keys = ["ref", "none", "none_mmq", "whole", "whole_mmq", "sub", "sub_mmq", "joint_line"]
    lines = [f"{'case':34s} " + " ".join(f"{k:>10s}" for k in keys)]
    for r in rows:
        lines.append(f"{r['scene'] + ' ' + r['baseline']:34s} " + " ".join(f"{r[k]:10.1f}" for k in keys))
    med = {k: np.median([r[k] for r in rows]) for k in keys}
    lines.append(f"{'median (' + str(len(rows)) + ' cases)':34s} " + " ".join(f"{med[k]:10.1f}" for k in keys))
    for a_, b_ in (("none", "none_mmq"), ("whole", "whole_mmq"), ("sub", "sub_mmq")):
        dlt = np.array([r[b_] - r[a_] for r in rows])
        lines.append(f"{b_} - {a_}: median {np.median(dlt):+.2f} dB, min {dlt.min():+.2f}, max {dlt.max():+.2f}, "
                     f"better by > 1 dB in {int((dlt < -1).sum())}, worse by > 1 dB in {int((dlt > 1).sum())}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    for s in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    jobs = SF.jobs()
    from multiprocessing import Pool
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            rows.append(r); print(f"{r['scene']:20s} {r['baseline']:7s} whole {r['whole']:6.1f} -> {r['whole_mmq']:6.1f}  "
                                  f"sub {r['sub']:6.1f} -> {r['sub_mmq']:6.1f}  none {r['none']:6.1f} -> {r['none_mmq']:6.1f}", flush=True)
    order = {j: i for i, j in enumerate(jobs)}
    rows.sort(key=lambda r: order[(r["scene"], r["baseline"])])
    txt = table(rows); open(os.path.join(OUT, "tabela.txt"), "w").write(txt + "\n"); print(txt)
    from joint_sweep_summary import bundle
    pngs = [os.path.join(OUT, "pages", f"{sc}__{tg}.png") for sc, tg in PAGES]
    print("wrote", bundle(pngs, os.path.join(OUT, "sata_mmq.pdf"), [(f"{sc} | {tg}", i, 0) for i, (sc, tg) in enumerate(PAGES)]))


if __name__ == "__main__":
    main()
