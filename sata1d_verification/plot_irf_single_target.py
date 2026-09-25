"""
IRF of ONE point target after multichannel reconstruction, with the new SATA
(delta_C0 map only where the target appears in the STFT).

Rows: target height 240 m (top) and 400 m (bottom).
Columns: left  = reference (monostatic), no SATA, SATA whole band
         right = reference (monostatic), no SATA, SATA per sub-band
Geometry: sar_recon make_topo_config -> Nrx = 4, PRF = 2000 Hz, bat step
100 m, bxt = [-225, -75, 75, 225] m; target at azimuth 0 on the iso-range
surface of r0 = 766.2 km.
Produces irf_single_target.png (+ prints the worst ambiguity of every curve,
new map and legacy "hold" map).
"""
import os, sys, io, tempfile, dataclasses, contextlib
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sar_reconstruction"))
import sar_recon as sar
import sar_recon.sata as sata_mod
import sar_recon.subband_recon as sub_mod
from sar_recon.config import make_topo_config
from sar_recon.signal_model import getRawData1D
from sar_recon.sata import sata_channels
from sar_recon.subband_sata_explicit import reconstruct_subband_explicit

HEIGHTS, DXT, SATA_OSF, AMB_MASK = (240.0, 400.0), 150.0, 4, 40


def case(h):
    cfg = make_topo_config(4, tempfile.mkdtemp(), "single", dxt=DXT)
    sc = cfg.scene
    dy = np.sqrt(sc.r0**2 - (sc.H - h)**2) - sc.y0
    cfg = dataclasses.replace(cfg, scene=dataclasses.replace(sc, extra_offsets=((0.0, dy, h),)))
    tr = sar.build_platform_tracks(cfg)
    raw = lambda pts, prx, vrx: getRawData1D(pts, tr.ptx, prx, tr.vtx, vrx, cfg.ta, cfg.sq_tx, cfg.sq_tx,
                                             cfg.theta_tx, cfg.theta_tx, cfg.system.wl, cfg.prf)
    tgt = cfg.scene.points[1:]
    sref1 = raw(cfg.scene.ptg[None, :], tr.ptx, tr.vtx)             # matched filter (1 point)
    sig = raw(tgt, tr.ptx, tr.vtx)                                   # ideal monostatic, full PRF
    s_ch = np.array([raw(tgt, tr.prx[i], tr.vrx[i])[::cfg.Nrx] for i in range(cfg.Nrx)])
    return cfg, tr, sref1, sig, s_ch


def reconstruct_all(cfg, tr, s_ch):
    with contextlib.redirect_stdout(io.StringIO()):
        return {"none": sar.reconstruct(cfg, tr, s_ch.copy()),
                "whole": sar.reconstruct(cfg, tr, sata_channels(cfg, tr, s_ch.copy(), sata_osf=SATA_OSF)),
                "sub": reconstruct_subband_explicit(cfg, tr, s_ch.copy(), use_sata=True,
                                                    sata_osf=SATA_OSF, correct_terms=("C0",))}


@contextlib.contextmanager
def legacy_maps():
    """Temporarily make both map builders use the legacy 'hold' mode."""
    a, b = sata_mod.build_delta_C0_array, sub_mod.build_delta_term_subband_array
    import sar_recon.subband_sata_explicit as ex
    c = ex.build_delta_term_subband_array
    sata_mod.build_delta_C0_array = lambda *p, **k: a(*p, **{**k, "mode": "hold"})
    hold_sub = lambda *p, **k: b(*p, **{**k, "mode": "hold"})
    sub_mod.build_delta_term_subband_array = hold_sub; ex.build_delta_term_subband_array = hold_sub
    try:
        yield
    finally:
        sata_mod.build_delta_C0_array, sub_mod.build_delta_term_subband_array = a, b
        ex.build_delta_term_subband_array = c


plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5})
fig, A = plt.subplots(2, 2, figsize=(16, 9.5), sharey=True)
for row, h in enumerate(HEIGHTS):
    cfg, tr, sref1, sig, s_ch = case(h)
    Na = cfg.Na
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref1))), Na // 2)
    rec = reconstruct_all(cfg, tr, s_ch)
    with legacy_maps():
        with contextlib.redirect_stdout(io.StringIO()):
            old = {"whole": sar.reconstruct(cfg, tr, sata_mod.sata_channels(cfg, tr, s_ch.copy(), sata_osf=SATA_OSF)),
                   "sub": reconstruct_subband_explicit(cfg, tr, s_ch.copy(), use_sata=True,
                                                       sata_osf=SATA_OSF, correct_terms=("C0",))}
    foc = {"ref": F(sig), **{k: F(v) for k, v in rec.items()}}
    foc_old = {k: F(v) for k, v in old.items()}
    pk = np.abs(foc["ref"]).max()
    db = {k: 20*np.log10(np.abs(v)/pk + 1e-12) for k, v in foc.items()}
    ipk = int(np.argmax(np.abs(foc["ref"])))
    mask = np.ones(Na, bool); mask[max(0, ipk-AMB_MASK):ipk+AMB_MASK+1] = False
    amb = lambda v: 20*np.log10(np.abs(v)[mask].max()/pk)
    A_ = {k: amb(v) for k, v in foc.items()}; A_old = {k: amb(v) for k, v in foc_old.items()}
    print(f"h = {h:.0f} m | worst ambiguity [dB]: ref {A_['ref']:.1f}, no SATA {A_['none']:.1f}, "
          f"SATA whole {A_['whole']:.1f} (legacy {A_old['whole']:.1f}), "
          f"SATA sub {A_['sub']:.1f} (legacy {A_old['sub']:.1f})")
    t = (np.arange(Na) - Na // 2) / cfg.prf
    for col, (key, lab, c) in enumerate((("whole", "SATA whole band", "tab:blue"),
                                         ("sub", "SATA per sub-band", "tab:green"))):
        ax = A[row, col]
        ax.plot(t, db["ref"], color="0.25", lw=1.0, label=f"ref (monostatic, amb {A_['ref']:.1f} dB)")
        ax.plot(t, db["none"], color="tab:red", lw=1.0, label=f"no SATA (amb {A_['none']:.1f} dB)")
        ax.plot(t, db[key], color=c, lw=1.0, label=f"{lab} (amb {A_[key]:.1f} dB)")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-100, 3); ax.grid(alpha=.3)
        ax.set_xlabel("Time [s]"); ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"h = {h:.0f} m | ref / no SATA / {lab} (new delta_C0 map)")
    A[row, 0].set_ylabel("[dB]")
dbat = cfg.array.bat[1] - cfg.array.bat[0]
fig.suptitle(f"IRF, one target at azimuth 0 | Nrx={cfg.Nrx} | PRF={cfg.prf:.0f} Hz | "
             f"$B_a$={cfg.abw:.1f} Hz | $\\Delta b_{{at}}$={dbat:.0f} m | "
             f"$b_{{xt}}$ = [{', '.join(f'{b:.0f}' for b in cfg.array.bxt)}] m", fontsize=12)
plt.tight_layout(); plt.savefig("irf_single_target.png", dpi=110); plt.close()
print("saved irf_single_target.png")
