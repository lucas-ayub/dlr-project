"""
Runs the example of run_example.py step by step, calling the same functions, and saves the numbers (numbers.json)
and one figure per step in plots/pipeline_doc/ (for the didactic PDF of the pipeline).
"""
import json, os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joint1d as J
from joint1d.config import ArrayGeometry, integration_time
from joint1d.reconstruction import GetCoeffNu, filter_matrix, GetInversionFilters
from joint1d.joint_line import LineModel, height_table, line_positions

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "pipeline_doc")
CH = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
num = {}
def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name + ".png"), dpi=130); plt.close(fig)

# ------------------------------------------------------------- 1) configuration
terrain = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
XS = np.array([-5200.0, -2600.0, 0.0, 2600.0, 5200.0])
BXT = np.array([0.0, 134.9, 20.5, 8.3])
cfg = J.make_config(list(zip(XS, terrain(XS))), nrx=len(BXT), bxt_max=0.0, tx_rx_channel0=True)
cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=BXT))
sp, sc = cfg.system, cfg.scene
num["cfg"] = dict(wl=sp.wl, vs=sp.vs, ve=sp.ve, da=sp.da, La=sp.La, abw=sp.abw, theta_deg=float(np.degrees(sp.theta)),
                  rDelay=sc.rDelay, r0=sc.r0, H=sc.H, y0=sc.y0, x0=sc.x0, h0=sc.h0, prf=cfg.prf, PRF_op=cfg.PRF_op,
                  Na=cfg.Na, Na_ch=cfg.Na_ch, T_ill=float(integration_time(sp, sc)), ta0=float(cfg.ta[0]), ta1=float(cfg.ta[-1]),
                  bat=cfg.array.bat.tolist(), bxt=cfg.array.bxt.tolist(), d=sp.vs / cfg.prf,
                  look_deg=float(np.degrees(np.arccos(sc.H / sc.r0))))
num["targets"] = [dict(x=float(x), h=float(h), dy=float(o[1]), pos=(sc.ptg + np.array(o)).tolist())
                  for x, h, o in zip(XS, terrain(XS), sc.extra_offsets)]

# ------------------------------------------------------------- 2) tracks
tr = J.build_platform_tracks(cfg)
k0 = cfg.Na // 2
num["tracks"] = dict(t=float(cfg.ta[k0]), tx=tr.ptx[k0].tolist(), rx=[tr.prx[i, k0].tolist() for i in range(cfg.Nrx)])

# ------------------------------------------------------------- 3) the echo of one target (inside getRawData1D)
p = sc.points[3]                                   # target at x = 0 (top of the mountain)
RT = np.linalg.norm(tr.ptx - p, axis=1); RR1 = np.linalg.norm(tr.prx[1] - p, axis=1)
sqT = np.arcsin(np.gradient(RT, 1 / cfg.prf) / sp.vs)
win = np.abs(sqT) <= sp.theta / 2
t = cfg.ta
fd = -np.gradient(2 * RT, 1 / cfg.prf) / sp.wl
num["echo"] = dict(Rmin=float(RT.min()), t_lit=float(win.sum() / cfg.prf), n_lit=int(win.sum()),
                   fd_max=float(np.abs(fd[win]).max()), RR1_minus_RT_at_t0=float(RR1[k0] - RT[k0]))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(t, RT - RT.min(), color="k", label="$R_T(t)-R_{min}$ (Tx)")
ax[0].plot(t, RR1 - RR1.min(), color=CH[1], ls="--", label="$R_{R,1}(t)-\\min R_{R,1}$ (Rx 1)")
ax[0].axvspan(t[win][0], t[win][-1], color="gold", alpha=0.25, label="antenna window (lit)")
ax[0].set_xlim(-2, 2); ax[0].set_ylim(-5, 300); ax[0].set_xlabel("azimuth time t [s]"); ax[0].set_ylabel("[m]")
ax[0].set_title("range history of the target at x = 0 (h = 1000 m)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
e = J.generate_reference(cfg, tr, p[None, :])
ax[1].plot(t, e.real, lw=0.4, color="k"); ax[1].set_xlim(-1.3, 1.3)
ax[1].set_title("its echo (monostatic, full PRF): Re{exp(-j2$\\pi$ 2R$_T$/$\\lambda$)} $\\cdot$ window"); ax[1].set_xlabel("t [s]")
ax[2].plot(t[win], fd[win], color="k"); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("instantaneous Doppler [Hz]")
ax[2].set_title("Doppler along the echo: $f_D=-\\frac{1}{\\lambda}\\frac{d(2R_T)}{dt}$"); ax[2].grid(alpha=.3)
save(fig, "s3_echo")

# ------------------------------------------------------------- 4) channels and reference
ch = J.generate_channels(cfg, tr)
ref = J.generate_reference(cfg, tr)
num["channels"] = dict(shape=list(ch.shape), ref_shape=list(ref.shape))
fig, ax = plt.subplots(1, 2, figsize=(16, 4.4))
tch = t[::cfg.Nrx]
for i in range(cfg.Nrx):
    ax[0].plot(tch, np.abs(ch[i]) + 6 * (3 - i), color=CH[i], lw=0.6)
    ax[0].text(tch[0] + 0.05, 6 * (3 - i) + 4.3, f"channel {i}: |s$_{i}$[k]| (b$_{{xt}}$ = {BXT[i]:g} m)", color=CH[i], fontsize=8)
ax[0].set_yticks([]); ax[0].set_xlabel("time of the channel samples [s] (500 Hz)"); ax[0].set_title("generate_channels: the 4 channels [4, 3072]")
f2000 = np.fft.fftshift(np.fft.fftfreq(cfg.Na, 1 / cfg.prf)); f500 = np.fft.fftshift(np.fft.fftfreq(cfg.Na_ch, 1 / cfg.PRF_op))
R = np.abs(np.fft.fftshift(np.fft.fft(ref))); S1 = np.abs(np.fft.fftshift(np.fft.fft(ch[1])))
ax[1].plot(f2000, 20 * np.log10(R / R.max()), color="tab:green", lw=0.8, label="Reference (monostatic, 2000 Hz)")
ax[1].plot(f500, 20 * np.log10(S1 / S1.max()), color=CH[1], lw=0.8, label="channel 1 (500 Hz): the band folded 3 times")
ax[1].axvspan(-250, 250, color=CH[1], alpha=0.08); ax[1].set_xlim(-1000, 1000); ax[1].set_ylim(-50, 3)
ax[1].set_xlabel("Doppler frequency [Hz]"); ax[1].set_ylabel("[dB]"); ax[1].legend(fontsize=8, loc="lower center"); ax[1].grid(alpha=.3)
ax[1].set_title("spectra: B$_a$ = 1235 Hz does not fit in 500 Hz -> aliasing")
save(fig, "s4_channels")

# ------------------------------------------------------------- 5) GetCoeffNu, H_i
coef = J.reference_coefficients(cfg, tr)
num["coef"] = [dict(C0=float(c[0]), C1=float(c[1]), C2=float(c[2]), Dt=float(c[3])) for c in coef]
C0, C1, C2, Dt, fit = GetCoeffNu(sc.ptg, tr.ptx, tr.prx[1], tr.vtx, tr.vrx[1], tr.ptx, tr.vtx, cfg.prf, sp.wl, cfg.ta,
                                 cfg.sq_tx, cfg.sq_rx[1], cfg.theta_tx, cfg.theta_rx[1], return_fit=True)
fig, ax = plt.subplots(1, 2, figsize=(16, 4.4))
ax[0].plot(fit["tau"], fit["d_range"], color=CH[1], lw=2, label="bistatic - monostatic range, channel 1")
a = fit["coeffs"]; ax[0].plot(fit["tau"], a[0] + a[1] * fit["tau"] + a[2] * fit["tau"] ** 2, "k--", lw=1, label="parabola fit")
ax[0].set_xlabel("time from closest approach [s]"); ax[0].set_ylabel("[m]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title("GetCoeffNu: fit of the range difference (reference point)")
fr = np.linspace(-cfg.abw / 2, cfg.abw / 2, 400)
for i, c in enumerate(coef):
    ph = -2 * np.pi * ((-c[1] + c[3]) * fr + c[2] * sp.wl * fr ** 2)
    ax[1].plot(fr, ph / (2 * np.pi), color=CH[i], label=f"channel {i}: $\\tau_i$ = {(c[3] - c[1]) * 1e3:.3f} ms, C$_0$ = {c[0]:.2f} m")
ax[1].set_xlabel("Doppler f [Hz]"); ax[1].set_ylabel("phase of H$_i$(f) without C$_0$ [turns]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("H$_i$(f) = exp{-j2$\\pi$[C$_0$/$\\lambda$ + ($-$C$_1$+D$_t$) f + C$_2$ $\\lambda$ f$^2$]}")
save(fig, "s5_coef")

# ------------------------------------------------------------- 6) standard reconstruction
nos = J.reconstruct(cfg, tr, ch)
fsub = -cfg.prf / 2 + np.arange(cfg.Na_ch) * cfg.PRF_op / cfg.Na_ch
hf = filter_matrix(cfg, tr, fsub)
cond = np.array([np.linalg.cond(hf[k]) for k in range(cfg.Na_ch)])
num["recon"] = dict(cond_median=float(np.median(cond)), cond_max=float(cond.max()), hf_shape=list(hf.shape))

# ------------------------------------------------------------- 7) line_maps
x = line_positions(cfg); h = terrain(x)
hg, tab = height_table(cfg, tr, h.min(), h.max())
dC0 = J.line_maps(cfg, tr, terrain)
fit_err = max(np.abs(np.polyval(np.polyfit(hg, tab[:, i], 3), hg) - tab[:, i]).max() for i in range(cfg.Nrx))
num["maps"] = dict(n_heights=len(hg), h_lo=float(hg[0]), h_hi=float(hg[-1]), dC0_at_1000=[float(np.interp(1000, hg, tab[:, i])) for i in range(cfg.Nrx)],
                   psi_at_peak=[float(-2 * np.pi * dC0[i].min() / sp.wl) if dC0[i].min() < 0 else float(2 * np.pi * dC0[i].max() / sp.wl) for i in range(cfg.Nrx)],
                   dC0_peak=[float(dC0[i][cfg.Na // 2]) for i in range(cfg.Nrx)], fit_err_m=float(fit_err), x_lo=float(x[0]), x_hi=float(x[-1]))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].fill_between(x / 1e3, h, color="#D7C9A7"); ax[0].plot(x / 1e3, h, color="#6D4C2F"); ax[0].set_xlim(-12, 12)
ax[0].set_xlabel("x = $\\ell$ d [km]"); ax[0].set_ylabel("h [m]"); ax[0].set_title("terrain(line_positions(cfg)): h at every place $\\ell$"); ax[0].grid(alpha=.3)
for i in range(cfg.Nrx):
    ax[1].plot(hg, tab[:, i] * 1e3, "o", ms=3, color=CH[i], label=f"channel {i}")
ax[1].set_xlabel("height h [m]"); ax[1].set_ylabel("$\\Delta C_0$ [mm]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("height_table: 33 heights -> residual_C0 (cubic fit through them)")
for i in range(cfg.Nrx):
    ax[2].plot(x / 1e3, 2 * np.pi * dC0[i] / sp.wl, color=CH[i], label=f"channel {i}")
ax[2].set_xlim(-12, 12); ax[2].set_xlabel("x [km]"); ax[2].set_ylabel("2$\\pi\\Delta C_0/\\lambda$ [rad]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
ax[2].set_title("line_maps: dC0[i, $\\ell$] -> phase of the screen")
save(fig, "s7_maps")

# ------------------------------------------------------------- 8) LineModel, sigma_max, CGLS
M = LineModel(cfg, tr, dC0)
smax = M.largest_singular_value(); damp2 = (0.03 * smax) ** 2
num["model"] = dict(K_nonzero=int((np.abs(generate := J.generate_reference(cfg, tr, sc.ptg[None, :])) > 0).sum()),
                    sigma_max=smax, mu=float(np.sqrt(damp2)), K_peak=float(np.abs(M.chirp).max()))
# CGLS written out (same as reconstruct_joint_line) to keep the history
s = ch.astype(complex); g = np.zeros(cfg.Na, complex); r = s.copy(); hist = []; n_it = 0
for ps in range(2):
    d = np.zeros(cfg.Na, complex); rp = r.copy(); z = M.adjoint(rp); pp = z.copy(); zz = z0 = np.vdot(z, z).real
    for _ in range(60):
        q = M.forward(pp); al = zz / (np.vdot(q, q).real + damp2 * np.vdot(pp, pp).real)
        d += al * pp; rp -= al * q; n_it += 1
        z = M.adjoint(rp) - damp2 * d; zn = np.vdot(z, z).real
        hist.append((n_it, ps + 1, float(np.linalg.norm(rp) / np.linalg.norm(s))))
        if zn <= 1e-8 * z0: break
        pp = z + (zn / zz) * pp; zz = zn
    g += d; r = rp
joint, info = J.reconstruct_joint_line(cfg, tr, ch, dC0, return_info=True)
U = np.fft.fft(g) * M.chirp; U[np.abs(np.fft.fftfreq(cfg.Na, 1 / cfg.prf)) > cfg.abw / 2] = 0
num["cgls"] = dict(iterations=n_it, unexplained_dB=float(20 * np.log10(info["unexplained"])),
                   same_as_function=float(np.abs(np.fft.ifft(U) - joint).max() / np.abs(joint).max()),
                   hist_every10=[hh for hh in hist if hh[0] in (1, 3, 10, 30, 60, 61, 90, 120)])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
fq = np.fft.fftshift(np.fft.fftfreq(cfg.Na, 1 / cfg.prf))
ax[0].plot(fq, np.abs(np.fft.fftshift(M.chirp)), color="k", lw=0.8); ax[0].set_xlim(-1000, 1000)
ax[0].set_xlabel("f [Hz]"); ax[0].set_title("LineModel: |K(f)| = |FFT(echo of the reference point)|"); ax[0].grid(alpha=.3)
for i in range(cfg.Nrx):
    ax[1].plot(x / 1e3, np.angle(np.fft.fftshift(M.screen[i])), color=CH[i], lw=0.8, label=f"channel {i}")
ax[1].set_xlim(-6, 6); ax[1].set_xlabel("x [km]"); ax[1].set_ylabel("angle [rad] (wrapped)"); ax[1].legend(fontsize=8)
ax[1].set_title("LineModel: screen[i] = exp(-j2$\\pi$ dC0/$\\lambda$)"); ax[1].grid(alpha=.3)
it_, ps_, res_ = np.array(hist).T
for ps in (1, 2):
    m = ps_ == ps; ax[2].semilogy(it_[m], res_[m], "-o", ms=2.5, label=f"pass {ps}")
ax[2].set_xlabel("iteration"); ax[2].set_ylabel("|s - A g| / |s|"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which="both")
ax[2].set_title(f"CGLS: {n_it} iterations, $\\mu$ = 0.03 $\\sigma_{{max}}$ = {np.sqrt(damp2):.2f}")
save(fig, "s8_model")

# ------------------------------------------------------------- 9) focusing and result
Hf = np.conj(np.fft.fft(J.generate_reference(cfg, tr, sc.ptg[None, :])))
focus = lambda v: np.abs(np.roll(np.fft.ifft(np.fft.fft(v) * Hf), cfg.Na // 2))
env = {"Reference": focus(ref), "no SATA": focus(nos), "joint final version": focus(joint)}
pk = env["Reference"].max(); tt = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
dt_amb = cfg.PRF_op * sp.wl * sc.r0 / (2 * sp.vs ** 2)
def amb(e):
    lobe = np.zeros(cfg.Na, bool); sel = np.zeros(cfg.Na, bool)
    for xx in XS:
        lobe |= np.abs(tt - xx / sp.vs) <= 0.02
        for m_ in (-2, -1, 1, 2): sel |= np.abs(tt - (xx / sp.vs + m_ * dt_amb)) <= 0.02
    sel &= ~lobe; return float(20 * np.log10(e[sel].max() / pk + 1e-15))
num["result"] = {k: amb(v) for k, v in env.items()}; num["result"]["dt_amb"] = dt_amb; num["result"]["X_amb"] = dt_amb * sp.vs
fig, ax = plt.subplots(2, 1, figsize=(16, 7.2), sharex=True)
db = lambda v: 20 * np.log10(v / pk + 1e-12)
ax[0].plot(tt, db(env["Reference"]), color="tab:green", lw=2.2, label=f"Reference ({num['result']['Reference']:.1f} dB)")
ax[0].plot(tt, db(env["joint final version"]), color="tab:purple", lw=1.1, label=f"joint final version ({num['result']['joint final version']:.1f} dB)")
ax[1].plot(tt, db(env["Reference"]), color="tab:green", lw=2.2, label=f"Reference ({num['result']['Reference']:.1f} dB)")
ax[1].plot(tt, db(env["no SATA"]), color="tab:red", lw=1.3, ls=(0, (5, 3)), label=f"no SATA ({num['result']['no SATA']:.1f} dB)")
for a_ in ax: a_.set_ylim(-90, 5); a_.grid(alpha=.3); a_.legend(fontsize=9, loc="upper left"); a_.set_ylabel("[dB]")
ax[1].set_xlim(tt[0], tt[-1]); ax[1].set_xlabel("azimuth time [s]")
ax[0].set_title("focused lines (FFT, x conj(K), IFFT); highest ambiguity in brackets")
save(fig, "s9_result")
json.dump(num, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
print(json.dumps(num, indent=1)[:4000])
