"""
IRF of one point target after reconstruction: no SATA / SATA whole band / SATA per sub-band.
Rows: target height 240 m and 400 m. Columns: whole band (left), per sub-band (right).
Array: Nrx = 4, PRF = 2000 Hz, bat on DPCA, bxt ~ U(0, 100) m (seed 0).
Output: irf_single_target.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sata1d as S

HEIGHTS = (240.0, 400.0)
BXT_MAX, SEED, SATA_OSF, AMB_MASK = 100.0, 0, 4, 40


def run(h):
    cfg = S.make_config([(0.0, h)], bxt_max=BXT_MAX, seed=SEED)
    tr = S.build_platform_tracks(cfg)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])   # matched filter (reference point)
    sig = S.generate_reference(cfg, tr)                            # ideal monostatic signal
    ch = S.generate_channels(cfg, tr)
    rec = {"none": S.reconstruct(cfg, tr, ch),
           "whole": S.reconstruct(cfg, tr, S.sata_channels(cfg, tr, ch, sata_osf=SATA_OSF)),
           "sub": S.reconstruct_subband(cfg, tr, ch, sata_osf=SATA_OSF)}
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref))), cfg.Na // 2)
    foc = {"ref": F(sig), **{k: F(v) for k, v in rec.items()}}
    return cfg, foc


def worst_ambiguity(foc):
    a = {k: np.abs(v) for k, v in foc.items()}
    pk, ipk = a["ref"].max(), int(np.argmax(a["ref"]))
    mask = np.ones(len(a["ref"]), bool)
    mask[max(0, ipk - AMB_MASK):ipk + AMB_MASK + 1] = False
    return {k: 20 * np.log10(v[mask].max() / pk) for k, v in a.items()}


fig, A = plt.subplots(2, 2, figsize=(16, 9.5), sharey=True)
for row, h in enumerate(HEIGHTS):
    cfg, foc = run(h)
    amb = worst_ambiguity(foc)
    print(f"h = {h:.0f} m | worst ambiguity [dB]: ref {amb['ref']:.1f}, no SATA {amb['none']:.1f}, "
          f"whole {amb['whole']:.1f}, sub {amb['sub']:.1f}")
    pk = np.abs(foc["ref"]).max()
    db = {k: 20 * np.log10(np.abs(v) / pk + 1e-12) for k, v in foc.items()}
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    for col, (key, lab, c) in enumerate((("whole", "SATA whole band", "tab:blue"),
                                         ("sub", "SATA per sub-band", "tab:green"))):
        ax = A[row, col]
        ax.plot(t, db["ref"], color="0.25", lw=1.0, label=f"reference (amb {amb['ref']:.1f} dB)")
        ax.plot(t, db["none"], color="tab:red", lw=1.0, label=f"no SATA (amb {amb['none']:.1f} dB)")
        ax.plot(t, db[key], color=c, lw=1.0, label=f"{lab} (amb {amb[key]:.1f} dB)")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-100, 3); ax.grid(alpha=.3)
        ax.set_xlabel("Time [s]"); ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"h = {h:.0f} m | reference / no SATA / {lab}")
    A[row, 0].set_ylabel("[dB]")
fig.suptitle(f"IRF, one target at azimuth 0 | Nrx={cfg.Nrx} | PRF={cfg.prf:.0f} Hz | "
             f"$B_a$={cfg.abw:.1f} Hz | $\\Delta b_{{at}}$={cfg.array.bat[1]:.2f} m (DPCA) | "
             f"$b_{{xt}}$ = [{', '.join(f'{b:.1f}' for b in cfg.array.bxt)}] m", fontsize=12)
plt.tight_layout(); plt.savefig("irf_single_target.png", dpi=110)
print("saved irf_single_target.png")
