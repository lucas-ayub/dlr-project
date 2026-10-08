"""
Figures of the review note on the multichannel reconstruction (reference point, point targets,
reconstruction steps). Channel 0 is the transmitter (b_at = b_xt = 0). Writes in --out:

    channel_vs_reconstruction.png   one channel focused alone (PRF/Nrx) against the reconstruction
    range_fit.png                   bistatic-minus-monostatic range of the reference point and its fit
    isolated_effects.png            IRF with cross-track baseline only, topography only, and both

    python reconstruction_review_figures.py --out ../results/sata1d/figures/reconstruction_review
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S
import tx_rx_channel as T
from tx_rx_channel_cases import make_case

BXT = [0.0, 100.0, 180.0, 300.0]


def channel_vs_reconstruction(png):
    """Target on the reference point: channel 0 alone (sampled at PRF/Nrx) shows the copies of the
    target; the reconstruction from the Nrx channels removes them."""
    cfg = make_case([(0.0, 0.0)], BXT); tr = S.build_platform_tracks(cfg)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    F = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * np.conj(np.fft.fft(sref))), cfg.Na // 2))
    ch = S.generate_channels(cfg, tr)
    alone = np.zeros(cfg.Na, complex); alone[::cfg.Nrx] = cfg.Nrx * ch[0]      # channel 0 on the full-PRF grid, zeros in between
    ref, one, rec = F(S.generate_reference(cfg, tr)), F(alone), F(S.reconstruct(cfg, tr, ch))
    pk = ref.max(); t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    db = lambda v: 20 * np.log10(v / pk + 1e-12)
    dt = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2 * cfg.system.vs ** 2)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    axs[0].plot(t, db(one), color="tab:blue", lw=0.8, label=f"one channel alone, sampled at {cfg.PRF_op:.0f} Hz")
    axs[0].plot(t, db(ref), color="0.2", lw=0.8, label="reference: monostatic at the transmitter, full PRF")
    axs[1].plot(t, db(rec), color="tab:red", lw=0.8, label=f"reconstruction from the {cfg.Nrx} channels, no SATA")
    axs[1].plot(t, db(ref), color="0.2", lw=0.8, label="reference: monostatic at the transmitter, full PRF")
    for m in (-2, -1, 1, 2):
        axs[0].annotate(f"{m * dt:+.2f} s", (m * dt, db(one)[np.argmin(np.abs(t - m * dt))]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8.5, color="tab:blue")
    axs[0].set_title("one channel focused alone: copies of the target", fontsize=10.5)
    axs[1].set_title("after the reconstruction: the copies cancel", fontsize=10.5)
    for ax in axs:
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-100, 12); ax.grid(alpha=0.3); ax.set_xlabel("azimuth time [s]")
        ax.legend(fontsize=8.5, loc="lower center")
    axs[0].set_ylabel("amplitude [dB re. reference peak]")
    fig.tight_layout(); fig.savefig(png, dpi=200); plt.close(fig)
    lv = {m: float(db(one)[np.abs(t - m * dt) <= 0.01].max()) for m in (-2, -1, 0, 1, 2)}
    print("channel alone, peaks at m*dt [dB]:", {k: round(v, 2) for k, v in lv.items()}, " dt =", round(dt, 4))


def range_fit(png):
    cfg = make_case([(0.0, 0.0)], BXT); tr = S.build_platform_tracks(cfg)
    fits = T.range_fits(cfg, tr)
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.0), sharex=True)
    for i, ax in enumerate(axs.ravel()):
        T.draw_range_fit(ax, cfg, i, fits[i], small=True)
    for ax in axs[-1]:
        ax.set_xlabel("$\\tau$ = time from the closest approach [s]", fontsize=9)
    fig.legend(handles=T.range_fit_legend_handles(), loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(png, dpi=200); plt.close(fig)
    for i, f in enumerate(fits):
        print(f"ch {i}: a0 = {f['a'][0]:.6f}  C0 = {f['C0']:.6f}  C1 = {f['C1']:.3e}  C2 = {f['C2']:.3e}  Dt = {1e3 * f['Dt']:.3f} ms")


def isolated_effects(png):
    cases = [("cross-track baseline only", 0.0, BXT), ("topography only", 1000.0, [0.0, 0.0, 0.0, 0.0]),
             ("both together", 1000.0, [0.0, 10.0, 18.0, 30.0]), ("both together", 200.0, BXT)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, sharey=True)
    for ax, (name, h, bxt) in zip(axs.ravel(), cases):
        cfg = make_case([(0.0, h)], bxt); tr = S.build_platform_tracks(cfg)
        irf = T.compute_irf(cfg, tr)
        T.draw_irf_line(ax, cfg, irf, small=True)
        ax.set_title(f"{name}: h = {h:.0f} m, $b_{{xt}}$ = [" + ", ".join(f"{b:g}" for b in bxt) + "] m", fontsize=10)
        ax.legend(*[x[::-1] for x in ax.get_legend_handles_labels()], fontsize=7.8, loc="upper center")
        ax.set_ylim(-100, 14)
        print(f"{name:28s} h = {h:6.0f}  bxt = {bxt}  main-lobe peak {irf['peak']:+.2f} dB")
    for ax in axs[0]:
        ax.set_xlabel("")
    for ax in axs[:, 1]:
        ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(png, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reconstruction_review")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    channel_vs_reconstruction(os.path.join(a.out, "channel_vs_reconstruction.png"))
    range_fit(os.path.join(a.out, "range_fit.png"))
    isolated_effects(os.path.join(a.out, "isolated_effects.png"))
    print("wrote 3 figures in", a.out)


if __name__ == "__main__":
    main()
