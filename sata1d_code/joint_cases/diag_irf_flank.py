"""Why the joint IRF of one_flank N8b100 has a broad pedestal: convergence / regularisation / model (diagnostic)."""
import sys, os, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from dataclasses import replace
import sata1d as S
from sata1d.config import ArrayGeometry
from sata1d.joint_line import height_table, line_maps, reconstruct_joint_line
import audit_dem as AD, sweep_final as SF

def setup(terrain, xs, hs, amps, tag="N8b100", bxt=None):
    bxt = AD.baseline(tag)[0] if bxt is None else np.asarray(bxt, float)
    cfg = S.make_config(list(zip(xs, hs)), nrx=len(bxt), bxt_max=0.0, amplitudes=amps, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg); ch = S.generate_channels(cfg, tr)
    half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
    hg = terrain(np.linspace(-half - 100, half + 100, 8001))
    tab = height_table(cfg, tr, hg.min(), hg.max())
    M = line_maps(cfg, tr, terrain, tab)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]); Hf = np.conj(np.fft.fft(sref))
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    ref = F(S.generate_reference(cfg, tr)); pk = ref.max()
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2 * cfg.system.vs ** 2)
    amb = lambda e: float(AD.ambiguity_level(t, e, pk, xs, cfg.system.vs, dt, orders=SF.orders_of(cfg)))
    t0 = xs[0] / cfg.system.vs
    ped = lambda e: float(np.median(20 * np.log10(e[(np.abs(t - t0) > 0.05) & (np.abs(t - t0) < 1.5)] / pk + 1e-15)))
    return cfg, tr, ch, M, F, ref, amb, ped

c = AD.SCENES["one_flank"][1](); xs, hs, terrain = c["xs"], c["hs"], c["terrain"]
out = {}
which = sys.argv[1]
if which in ("base", "it300", "reg01", "reg003"):
    cfg, tr, ch, M, F, ref, amb, ped = setup(terrain, xs, hs, c["amps"])
    kw = dict(base={}, it300=dict(iterations=300, tol=1e-7), reg01=dict(reg=0.01, iterations=300, tol=1e-7),
              reg003=dict(reg=0.003, iterations=300, tol=1e-7))[which]
elif which in ("plateau", "plateau_it300"):           # same target height everywhere: no slope, same dC0 at the target
    h0 = float(hs[0]); T = lambda x: np.full_like(np.asarray(x, float), h0)
    cfg, tr, ch, M, F, ref, amb, ped = setup(T, xs, hs, c["amps"])
    kw = {} if which == "plateau" else dict(iterations=300, tol=1e-7)
elif which == "zeroh":                                # the target at h = 0 on flat terrain (no topography at all)
    T = lambda x: np.zeros_like(np.asarray(x, float))
    cfg, tr, ch, M, F, ref, amb, ped = setup(T, xs, 0 * hs, c["amps"])
    kw = {}
elif which in ("zeroh_bxt0", "zeroh_N4b100", "zeroh_N8b500", "zeroh_N2b100"):
    T = lambda x: np.zeros_like(np.asarray(x, float))
    tag = {"zeroh_bxt0": "N8b100", "zeroh_N4b100": "N4b100", "zeroh_N8b500": "N8b500", "zeroh_N2b100": "N2b100"}[which]
    cfg, tr, ch, M, F, ref, amb, ped = setup(T, xs, 0 * hs, c["amps"], tag=tag, bxt=np.zeros(8) if which == "zeroh_bxt0" else None)
    kw = {}
t1 = time.time()
y, info = reconstruct_joint_line(cfg, tr, ch, M, return_info=True, **kw)
e = F(y); none = F(S.reconstruct(cfg, tr, ch))
r = dict(case=which, h=float(hs[0]) if which != "zeroh" else 0.0, ref=amb(ref), none=amb(none), joint=amb(e), pedestal_joint=ped(e), pedestal_ref=ped(ref),
         iterations=info.get("iterations"), unexplained_dB=float(20 * np.log10(info["unexplained"])), sec=time.time() - t1)
print(json.dumps(r))
json.dump(r, open(os.path.join(HERE, f"diag_flank_{which}.json"), "w"))
