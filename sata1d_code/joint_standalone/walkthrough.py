"""
The path of the data through the joint, step by step, on the example of run_example.py (5 targets on a 1000 m mountain,
bxt = [0, 134.9, 20.5, 8.3] m). Writes plots/walkthrough/joint_walkthrough.pdf (and one PNG per page).

The CGLS loop below is the loop of reconstruct_joint_line written out, keeping the line after chosen iterations; the
script checks that its result is the same as reconstruct_joint_line.
"""
import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import joint1d as J
from joint1d.config import ArrayGeometry
from joint1d.joint_line import LineModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "walkthrough")
os.makedirs(OUT, exist_ok=True)
CH = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

terrain = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
XS = np.array([-5200.0, -2600.0, 0.0, 2600.0, 5200.0])
BXT = np.array([0.0, 134.9, 20.5, 8.3])
cfg = J.make_config(list(zip(XS, terrain(XS))), nrx=4, bxt_max=0.0, tx_rx_channel0=True)
cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=BXT))
tr = J.build_platform_tracks(cfg)
N, Nrx, prf, d = cfg.Na, cfg.Nrx, cfg.prf, cfg.system.vs / cfg.prf
x_km = (np.arange(N) - N // 2) * d / 1e3                       # natural order of the line
t_ch = (np.arange(cfg.Na_ch) - cfg.Na_ch // 2) / cfg.PRF_op     # time of the channel samples
nat = np.fft.fftshift                                           # lag order (code) -> natural order (plots)

# ---------------------------------------------------------------- inputs
s = J.generate_channels(cfg, tr)                                # what the receivers recorded
dC0 = J.line_maps(cfg, tr, terrain)                             # from the DEM only
M = LineModel(cfg, tr, dC0)                                     # K, K_i, screens
reference = J.generate_reference(cfg, tr)
no_sata = J.reconstruct(cfg, tr, s)

# ---------------------------------------------------------------- the CGLS of reconstruct_joint_line, written out
reg, passes, iterations, tol = 0.03, 2, 60, 1e-4
smax = M.largest_singular_value()
damp2 = (reg * smax) ** 2
g = np.zeros(N, complex); r = s.copy(); n_it = 0
keep = {}; hist = []
KEEP = (1, 3, 10, 30)
for ps in range(passes):
    dlt = np.zeros(N, complex); rp = r.copy()
    z = M.adjoint(rp); p = z.copy(); zz = z0 = np.vdot(z, z).real
    for _ in range(iterations):
        q = M.forward(p)
        alpha = zz / (np.vdot(q, q).real + damp2 * np.vdot(p, p).real)
        dlt += alpha * p; rp -= alpha * q; n_it += 1
        z = M.adjoint(rp) - damp2 * dlt
        zz_new = np.vdot(z, z).real
        hist.append((n_it, ps + 1, np.linalg.norm(rp) / np.linalg.norm(s)))
        if n_it in KEEP:
            keep[n_it] = g + dlt
        if zz_new <= tol ** 2 * z0:
            break
        p = z + (zz_new / zz) * p; zz = zz_new
    g += dlt; r = rp
keep[n_it] = g.copy()


def raw_from_g(gg):
    U = np.fft.fft(gg) * M.chirp
    U[np.abs(np.fft.fftfreq(N, 1 / prf)) > cfg.abw / 2] = 0
    return np.fft.ifft(U)


joint = raw_from_g(g)
check = J.reconstruct_joint_line(cfg, tr, s, dC0)
print("same as reconstruct_joint_line:", float(np.abs(joint - check).max() / np.abs(check).max()))

Hf = np.conj(np.fft.fft(J.generate_reference(cfg, tr, cfg.scene.ptg[None, :])))
focus = lambda v: np.abs(np.roll(np.fft.ifft(np.fft.fft(v) * Hf), N // 2))
pk = focus(reference).max()
dB = lambda v: 20 * np.log10(v / pk + 1e-12)
t_full = (np.arange(N) - N // 2) / prf


def amb(e):
    vs = cfg.system.vs; dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2 * vs ** 2)
    lobe = np.zeros(N, bool); sel = np.zeros(N, bool)
    for x in XS:
        lobe |= np.abs(t_full - x / vs) <= 0.02
        for m in (-2, -1, 1, 2):
            sel |= np.abs(t_full - (x / vs + m * dt)) <= 0.02
    sel &= ~lobe
    return 20 * np.log10(e[sel].max() / pk + 1e-15)


foot = (f"Example of run_example.py: 5 targets on a 1000 m mountain, $N_{{rx}}$ = 4, PRF = {prf:.0f} Hz, PRF$_{{op}}$ = {cfg.PRF_op:.0f} Hz, "
        f"$b_{{xt}}$ = [0, 134.9, 20.5, 8.3] m, channel 0 on the transmitter. The joint receives only the channels, the system geometry and the DEM.")
pages = []

# ================================================================ page 1: inputs
fig, ax = plt.subplots(2, 2, figsize=(20, 10.5))
fig.suptitle("Step 0 — what the joint receives, and what it prepares once (P1–P3)", fontsize=14, y=0.985)
a = ax[0, 0]
for i in range(Nrx):
    a.plot(t_ch, np.abs(s[i]) + 6 * (Nrx - 1 - i), color=CH[i], lw=0.6)
    a.text(t_ch[0] + 0.05, 6 * (Nrx - 1 - i) + 4.4, f"channel {i}: $|s_{i}[k]|$", color=CH[i], fontsize=9)
a.set_yticks([]); a.set_xlabel("time of the channel samples [s] (one sample every 2 ms)")
a.set_title("INPUT 1 — the data: the 4 channels as recorded (500 Hz, aliased; the targets are not visible yet)", fontsize=10)
a.grid(alpha=0.3)
a = ax[0, 1]
xx = np.linspace(-12e3, 12e3, 2001)
a.fill_between(xx / 1e3, terrain(xx), color="#D7C9A7"); a.plot(xx / 1e3, terrain(xx), color="#6D4C2F")
a.set_xlabel("azimuth x [km]"); a.set_ylabel("h [m]"); a.grid(alpha=0.3)
a.set_title("INPUT 2 — the DEM h(x) (no target positions are given)", fontsize=10)
a = ax[1, 0]
f = np.fft.fftshift(np.fft.fftfreq(N, 1 / prf))
a.plot(f, np.abs(np.fft.fftshift(M.chirp)), color="k", lw=1.2, label="$|K(f)|$: echo of one point, full PRF (P2)")
a.set_ylabel("$|K(f)|$"); a.set_xlabel("Doppler frequency f [Hz]"); a.set_xlim(-1000, 1000); a.grid(alpha=0.3)
a2 = a.twinx()
for i in range(Nrx):
    ph = np.unwrap(np.angle(np.fft.fftshift(M.K[i] / M.chirp))) if i else np.zeros(N)
    ph = ph - ph[N // 2] if i else ph
    a2.plot(f, ph / (2 * np.pi), color=CH[i], lw=1.3, ls="--", label=f"phase of $H_{i}$ (P1)")
a2.set_ylabel("phase of $H_i(f)$ [turns] (dashed)")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center")
a.set_title("PREPARED 1 — $K_i(f) = K(f) H_i(f)$ (self.K): the echo of one point in each channel, in frequency", fontsize=10)
a = ax[1, 1]
for i in range(Nrx):
    a.plot(x_km, 2 * np.pi * dC0[i] / cfg.system.wl, color=CH[i], lw=1.4, label=f"channel {i}")
a.set_xlim(-12, 12); a.set_xlabel("azimuth x [km] (place $\\ell$ of the focused line)"); a.set_ylabel("$\\psi_i$ [rad]")
a.grid(alpha=0.3); a.legend(fontsize=9)
a.set_title("PREPARED 2 — the screens $e^{-j\\psi_i[\\ell]}$ (self.screen, P3): one phase per place, from the DEM", fontsize=10)
fig.text(0.5, 0.01, foot, ha="center", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); pages.append(fig)

# ================================================================ page 2: first look = adjoint
z_i = []
for i in range(Nrx):
    v = np.zeros(N, complex); v[::Nrx] = s[i]
    z_i.append(np.conj(M.screen[i]) * np.fft.ifft(np.fft.fft(v) * np.conj(M.K[i])))
z_i = np.array(z_i); zsum = z_i.sum(0)
fig, ax = plt.subplots(2, 1, figsize=(20, 10.5), sharex=True)
fig.suptitle("Step 1 — the adjoint $A^H s$: focus every channel with its own echo, remove the relief phase of every place, add the channels "
             "(this is the direction of the first CGLS step)", fontsize=13, y=0.985)
zmax = np.abs(zsum).max()
for i in range(Nrx):
    ax[0].plot(x_km, 20 * np.log10(np.abs(nat(z_i[i])) / zmax + 1e-12), color=CH[i], lw=0.7, label=f"channel {i} alone")
ax[0].set_title("each channel alone, focused (zeros between its samples -> FFT -> x conj($K_i$) -> IFFT -> x conj(screen)): "
                "the targets plus ghosts at $\\pm$6.2 km (the folds of 500 Hz)", fontsize=10)
ax[1].plot(x_km, 20 * np.log10(np.abs(nat(zsum)) / zmax + 1e-12), color="k", lw=0.8, label="sum of the 4 channels")
ax[1].set_title("sum of the channels: the targets add up in phase, the ghosts only partly cancel -> the iterations remove what is left", fontsize=10)
for a in ax:
    a.set_ylim(-60, 3); a.grid(alpha=0.3); a.legend(fontsize=9, loc="upper right"); a.set_ylabel("[dB]")
    for x in XS:
        a.axvline(x / 1e3, color="0.6", lw=0.6, ls=":")
ax[1].set_xlim(-16, 16); ax[1].set_xlabel("azimuth x [km] (place $\\ell$ of the focused line; dotted: the targets, not given to the joint)")
fig.text(0.5, 0.01, foot, ha="center", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); pages.append(fig)

# ================================================================ page 3: iterations
fig = plt.figure(figsize=(20, 10.5))
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
fig.suptitle("Step 2 — CGLS: each iteration applies the model (B) and the adjoint (C) once and corrects the line where the channels are not "
             "explained yet", fontsize=13, y=0.985)
a = fig.add_subplot(gs[:, 0])
ref_db = dB(focus(reference))
a.plot(t_full, ref_db, color="tab:green", lw=2.0, label=f"Reference ({amb(focus(reference)):.1f} dB)")
cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(keep)))
for (it, gk), c in zip(sorted(keep.items()), cols):
    e = focus(raw_from_g(gk))
    a.plot(t_full, dB(e), color=c, lw=0.9 if it < n_it else 1.3,
           label=f"joint after {it} iteration{'s' if it > 1 else ''}{' (final)' if it == n_it else ''} ({amb(e):.1f} dB)")
a.set_xlim(t_full[0], t_full[-1]); a.set_ylim(-80, 3); a.grid(alpha=0.3); a.legend(fontsize=9, loc="upper left")
a.set_xlabel("Azimuth time [s]"); a.set_ylabel("[dB]")
a.set_title("focused line after some iterations (highest ambiguity in brackets): the ghosts go down iteration by iteration", fontsize=10)
b = fig.add_subplot(gs[0, 1])
it_, ps_, res_ = np.array(hist).T
for ps in (1, 2):
    m = ps_ == ps
    b.semilogy(it_[m], res_[m], "-o", ms=2.5, label=f"pass {ps}")
b.set_xlabel("iteration"); b.set_ylabel("$\\|s - A\\hat g\\| / \\|s\\|$"); b.grid(alpha=0.3, which="both"); b.legend(fontsize=9)
b.set_title("part of the channels not explained by the current line", fontsize=10)
c2 = fig.add_subplot(gs[1, 1]); c2.axis("off")
c2.text(0.0, 1.0, "\n".join([
    "One iteration (CGLS):",
    "  q = A p                     model: screen, FFT, x K_i, IFFT, [::Nrx]",
    "  alpha = |z|^2 / (|q|^2 + mu^2 |p|^2)",
    "  line += alpha p             correct the line",
    "  residual -= alpha q         what is left in the channels",
    "  z = A^H residual - mu^2 line   adjoint: what to correct next",
    "  p = z + beta p              new direction",
    "",
    f"mu = 0.03 x sigma_max = {np.sqrt(damp2):.2f};  2 passes;",
    f"{n_it} iterations in total (stop: gradient down by 1e-4)",
]), va="top", family="monospace", fontsize=9.5)
fig.text(0.5, 0.01, foot, ha="center", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); pages.append(fig)

# ================================================================ page 4: output
fig, ax = plt.subplots(2, 1, figsize=(20, 10.5), sharex=True)
fig.suptitle("Step 3 — output: $U = K\\hat g$ (raw line at 2000 Hz, as the standard reconstruction gives), then the same focusing as every "
             "method (FFT, x $K^*$, IFFT)", fontsize=13, y=0.985)
er, ej, en = focus(reference), focus(joint), focus(no_sata)
ax[0].plot(t_full, dB(er), color="tab:green", lw=2.2, label=f"Reference ({amb(er):.1f} dB)")
ax[0].plot(t_full, dB(ej), color="tab:purple", lw=1.2, label=f"joint final version ({amb(ej):.1f} dB)")
ax[1].plot(t_full, dB(er), color="tab:green", lw=2.2, label=f"Reference ({amb(er):.1f} dB)")
ax[1].plot(t_full, dB(en), color="tab:red", lw=1.4, ls=(0, (5, 3)), label=f"no SATA ({amb(en):.1f} dB)")
for a in ax:
    a.set_ylim(-90, 5); a.grid(alpha=0.3); a.legend(fontsize=10, loc="upper left"); a.set_ylabel("[dB]")
ax[1].set_xlim(t_full[0], t_full[-1]); ax[1].set_xlabel("Azimuth time [s]")
ax[0].set_title("Reference and joint", fontsize=10); ax[1].set_title("Reference and no SATA (standard reconstruction, relief ignored)", fontsize=10)
fig.text(0.5, 0.01, foot, ha="center", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); pages.append(fig)

with PdfPages(os.path.join(OUT, "joint_walkthrough.pdf")) as pdf:
    for k, fg in enumerate(pages, 1):
        fg.savefig(os.path.join(OUT, f"step{k - 1}.png"), dpi=90)
        pdf.savefig(fg)
        plt.close(fg)
print("saved", os.path.join(OUT, "joint_walkthrough.pdf"), "| iterations", n_it, "| kept", sorted(keep))
