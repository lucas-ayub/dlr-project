"""No SATA, one target: worst ambiguity vs height, and the channel phase residuals."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S

hs = np.arange(0, 3001, 25.0)
amb, phis = [], []
for h in hs:
    c = S.make_config([(0.0, h)]); t = S.build_platform_tracks(c)
    sref = S.generate_reference(c, t, c.scene.ptg[None, :])
    F = lambda x: np.roll(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(sref))), c.Na // 2)
    ref = F(S.generate_reference(c, t)); rec = F(S.reconstruct(c, t, S.generate_channels(c, t)))
    m = np.ones(c.Na, bool); m[c.Na // 2 - 40:c.Na // 2 + 41] = False
    amb.append(20 * np.log10(np.abs(rec)[m].max() / np.abs(ref).max()))
    phis.append([np.degrees(2 * np.pi * S.residual_C0(c, t, c.scene.points[1], i) / c.system.wl) for i in range(c.Nrx)])
amb, phis = np.array(amb), np.array(phis)
np.savez("probe_nosata_vs_height.npz", hs=hs, amb=amb, phis=phis, bxt=c.array.bxt)
spread = np.degrees(np.abs(np.angle(np.exp(1j * np.radians(phis - phis.mean(1, keepdims=True))))).max(1))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
a1.plot(hs, amb, "-o", ms=3, color="tab:red"); a1.set_ylabel("worst ambiguity, no SATA [dB]"); a1.grid(alpha=.3)
a1.set_title("SATA 1D chain, one target, no SATA: worst ambiguity vs target height ($b_{xt}$ = %s m)"
             % ", ".join(f"{b:.1f}" for b in c.array.bxt), fontsize=10)
for i in range(c.Nrx):
    a2.plot(hs, np.mod(phis[:, i] + 180, 360) - 180, label=f"ch {i}, $b_{{xt}}$ = {c.array.bxt[i]:.1f} m")
a2.set_ylabel(r"residual phase $2\pi\Delta C_{0,i}/\lambda$ wrapped [deg]"); a2.set_xlabel("target height [m]")
a2.legend(fontsize=8); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig("probe_nosata_vs_height.png", dpi=130)
for h, a, p in zip(hs[::8], amb[::8], phis[::8]):
    print(f"h={h:5.0f}  amb {a:6.1f} dB  phases " + " ".join(f"{np.mod(x + 180, 360) - 180:7.1f}" for x in p))
