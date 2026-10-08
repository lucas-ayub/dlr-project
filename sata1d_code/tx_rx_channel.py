"""
Channel 0 as the transmitter: one satellite transmits and receives (b_at = b_xt = 0), the other
channels only receive (b_at on the DPCA condition, b_xt ~ U(0, bxt)). Three figures in --out:

    geometry.png     positions of the transmitter and of the receivers (along-track, cross-track)
    range_fit.png    per channel: bistatic-minus-monostatic range of the reference point against
                     time and its quadratic fit (the fit of GetCoeffNu), with the fit coefficients
    irf.png          IRF of one target after reconstruction against the monostatic reference

    python tx_rx_channel.py                      # target on the reference point (x = 0, h = 0)
    python tx_rx_channel.py --h 240              # elevated target, reconstruction without SATA
    python tx_rx_channel.py --bxt 300 --seed 3

The draw_* functions draw one block on given axes; tx_rx_channel_cases.py puts them in one figure per case.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sata1d as S

RX_COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown", "tab:pink", "tab:cyan"]
AMB_MASK = 40          # samples around the peak excluded from the highest-ambiguity search
ZOOM_OSF = 16          # oversampling of the main-lobe zoom


def is_tx(cfg, i):
    """Channel i sits on the transmitter."""
    return cfg.array.bat[i] == 0 and cfg.array.bxt[i] == 0


# ------------------------------------------------------------------ channel geometry
def draw_geometry(ax, cfg, small=False):
    """Transmitter and receivers in the (along-track, cross-track) plane. small: sizes for a figure block."""
    bat, bxt = cfg.array.bat, cfg.array.bxt
    ax.plot(0, 0, "s", ms=16 if small else 20, mfc="none", mec="k", mew=2.0, label="transmitter (TX)", zorder=3)
    for i in range(cfg.Nrx):
        ax.plot(-bat[i], bxt[i], "o", ms=9 if small else 11, color=RX_COLORS[i % len(RX_COLORS)], mec="k", zorder=4,
                label=f"receiver {i}: $b_{{at}}$ = {bat[i]:.2f} m, $b_{{xt}}$ = {bxt[i]:.2f} m")
        ax.annotate("TX = RX0" if is_tx(cfg, i) else f"RX{i}", (-bat[i], bxt[i]),
                    xytext=(12, 10), textcoords="offset points", fontsize=9.5 if small else 11)
    px = 0.12 * max(bat.max(), 1.0) + 1.0
    py = 0.12 * max(np.abs(bxt).max(), 1.0) + 1.0
    ax.set_xlim(-bat.max() - px, max(-bat.min(), 0) + 2.5 * px)
    ax.set_ylim(min(bxt.min(), 0) - py, max(bxt.max(), 0) + 2.0 * py)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.annotate("", xy=(x1 - 0.03 * (x1 - x0), y1 - 0.07 * (y1 - y0)), xytext=(x1 - 0.20 * (x1 - x0), y1 - 0.07 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="k"))
    ax.text(x1 - 0.115 * (x1 - x0), y1 - 0.05 * (y1 - y0), "flight direction", ha="center", va="bottom", fontsize=9 if small else 10)
    ax.axhline(0, color="0.6", lw=0.7, zorder=1); ax.axvline(0, color="0.6", lw=0.7, zorder=1)
    ax.set_xlabel("along-track position relative to the transmitter [m]")
    ax.set_ylabel("cross-track position relative to the transmitter [m]")
    ax.grid(alpha=0.3)
    if small:                                                             # legend under the axes: it never hides a receiver
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False)
    else:
        ax.legend(fontsize=9.5, loc="upper left")


def plot_geometry(cfg, png):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    draw_geometry(ax, cfg)
    ax.set_title(f"Channel geometry: {cfg.Nrx} receivers, channel 0 is the transmitter | "
                 f"along-track step {cfg.array.bat[1] - cfg.array.bat[0]:.2f} m (DPCA at PRF = {cfg.prf:.0f} Hz)", fontsize=11)
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ quadratic fit of the range difference
def range_fits(cfg, tr):
    """Per channel: the fitted bistatic-minus-monostatic range of the reference point (the fit of GetCoeffNu)
    and the coefficients the reconstruction derives from it. List of dict(tau, d, a, C0, C1, C2, Dt)."""
    fits = []
    for i in range(cfg.Nrx):
        C0, C1, C2, Dt, fit = S.GetCoeffNu(cfg.scene.ptg, tr.ptx, tr.prx[i], tr.vtx, tr.vrx[i], tr.ptx, tr.vtx,
                                           cfg.prf, cfg.system.wl, cfg.ta, cfg.sq_tx, cfg.sq_rx[i],
                                           cfg.theta_tx, cfg.theta_rx[i], return_fit=True)
        fits.append(dict(tau=fit["tau"], d=fit["d_range"], a=tuple(fit["coeffs"]), C0=C0, C1=C1, C2=C2, Dt=Dt))
    return fits


def draw_range_fit(ax, cfg, i, fit, small=False):
    """Range difference of channel i and its quadratic fit, with the coefficients in a box."""
    tau, d, (a0, a1, a2) = fit["tau"], fit["d"], fit["a"]
    step = max(1, len(tau) // (40 if small else 60))
    ax.plot(tau[::step], 1e3 * d[::step], "o", ms=4 if small else 5, mfc="none", mec=RX_COLORS[i % len(RX_COLORS)])
    ax.plot(tau, 1e3 * (a0 + a1 * tau + a2 * tau ** 2), "-", color="k", lw=1.3)
    if np.ptp(d) == 0:
        ax.set_ylim(1e3 * a0 - 1.0, 1e3 * a0 + 1.0)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    txt = ("fit: $\\Delta r(\\tau) = a_0 + a_1\\tau + a_2\\tau^2$\n"
           f"$a_0$ = {a0:.6f} m\n$a_1$ = {a1:.3e} m/s\n$a_2$ = {a2:.6e} m/s$^2$")
    if not small:
        txt += ("\ncoefficients used by the reconstruction:\n"
                f"$C_0$ = {fit['C0']:.6f} m, $C_1$ = {fit['C1']:.3e} s\n"
                f"$C_2$ = {fit['C2']:.3e} s$^2$/m, $\\Delta t$ = {1e3 * fit['Dt']:.3f} ms")
    ybox, va = (0.96, "top") if a2 >= 0 else (0.04, "bottom")            # away from the vertex of the parabola
    ax.text(0.5, ybox, txt, transform=ax.transAxes, ha="center", va=va, fontsize=7.6 if small else 9.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    tag = " (transmitter)" if is_tx(cfg, i) else ""
    ax.set_title(f"channel {i}{tag}: $b_{{at}}$ = {cfg.array.bat[i]:.2f} m, $b_{{xt}}$ = {cfg.array.bxt[i]:.2f} m",
                 fontsize=9.5 if small else 11)
    ax.set_ylabel("bistatic $-$ monostatic\nrange [mm]" if small else "bistatic $-$ monostatic range [mm]",
                  fontsize=8.5 if small else None)
    if small:
        ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.3)


def range_fit_legend_handles():
    return [Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec="0.3", label="range difference (simulated geometry)"),
            Line2D([], [], color="k", lw=1.3, label="quadratic fit")]


def plot_range_fit(cfg, tr, png):
    """Quadratic fit of the bistatic-minus-monostatic range of the reference point, per channel."""
    Nrx = cfg.Nrx
    nrow = int(np.ceil(Nrx / 2))
    fig, axs = plt.subplots(nrow, 2, figsize=(15, 4.6 * nrow), sharex=True, squeeze=False)
    fits = range_fits(cfg, tr)
    for i in range(Nrx):
        draw_range_fit(axs.ravel()[i], cfg, i, fits[i])
    for ax in axs.ravel()[Nrx:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("$\\tau$ = time from the closest approach [s]")
    fig.suptitle("Range difference of each channel to the transmitter's own (monostatic) range, reference point, and its quadratic fit\n"
                 "bistatic = transmitter $\\to$ point $\\to$ receiver;  monostatic = transmitter $\\to$ point $\\to$ transmitter", fontsize=12)
    fig.legend(handles=range_fit_legend_handles(), loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(png, dpi=130); plt.close(fig)
    return [(i, *f["a"], f["C0"], f["C1"], f["C2"], f["Dt"]) for i, f in enumerate(fits)]


# ------------------------------------------------------------------ IRF after reconstruction
def upsample(x, osf):
    """Band-limited interpolation by zero-padding the spectrum."""
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x))
    Y = np.zeros(N * osf, np.complex128)
    Y[(N * osf - N) // 2:(N * osf - N) // 2 + N] = X
    return np.fft.ifft(np.fft.ifftshift(Y)) * osf


def compute_irf(cfg, tr):
    """Focused monostatic reference and focused reconstruction (no SATA) of the scene's targets."""
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])          # matched filter: the reference point
    sig = S.generate_reference(cfg, tr)                                   # monostatic signal of the targets, full PRF
    rec = S.reconstruct(cfg, tr, S.generate_channels(cfg, tr))            # reconstruction from the Nrx channels
    F = lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * np.conj(np.fft.fft(sref))), cfg.Na // 2)
    foc = {"ref": F(sig), "rec": F(rec)}
    a = {k: np.abs(v) for k, v in foc.items()}
    pk, ipk = a["ref"].max(), int(np.argmax(a["ref"]))
    mask = np.ones(cfg.Na, bool); mask[max(0, ipk - AMB_MASK):ipk + AMB_MASK + 1] = False
    fa = np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)
    Sb = np.fft.fft(sig); Sb[np.abs(fa) > cfg.abw / 2] = 0                 # reference limited to the processed bandwidth
    foc_bl = F(np.fft.ifft(Sb))
    t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
    tu = (np.arange(cfg.Na * ZOOM_OSF) - (cfg.Na // 2) * ZOOM_OSF) / (cfg.prf * ZOOM_OSF)
    up = {k: np.abs(upsample(v, ZOOM_OSF)) for k, v in foc.items()}        # oversampled, for the main lobes
    targets = []                                                          # every target: where it focuses and its main-lobe peak
    for x, _, h in cfg.scene.extra_offsets:
        near = np.flatnonzero(np.abs(tu - x / cfg.system.vs) <= 5e-3)
        i0 = near[np.argmax(up["ref"][near])]
        lobe = np.abs(tu - tu[i0]) <= 0.5e-3
        targets.append(dict(x=x, h=h, t0=tu[i0], ref=up["ref"][i0],
                            peak=20 * np.log10(up["rec"][lobe].max() / up["ref"][i0])))   # at the target, re. its own reference peak
    return dict(t=t, tu=tu, up=up, foc=foc, a=a, pk=pk, ipk=ipk, targets=targets,
                amb={k: 20 * np.log10(v[mask].max() / pk) for k, v in a.items()},
                peak=max(targets, key=lambda q: q["ref"])["peak"],        # of the strongest target (the only one, with one target)
                peak_bl=20 * np.log10(np.abs(foc_bl).max() / pk),
                err=float(np.abs(foc["rec"] - foc_bl).max() / pk))


def irf_note(cfg, irf):
    return (f"The reconstruction keeps only the processed bandwidth $B_a$ = {cfg.abw:.0f} Hz; the reference is not band-limited. "
            f"The reference limited to $B_a$ peaks at {irf['peak_bl']:+.2f} dB.")


def draw_irf_line(a1, cfg, irf, small=False, peak_in_legend=True):
    """Whole azimuth line: reconstruction underneath, reference on top."""
    t, a, pk = irf["t"], irf["a"], irf["pk"]
    pks = [q["peak"] for q in irf["targets"]]                             # one per target, in the order of the targets
    note = "" if not peak_in_legend or len(pks) > 6 else (
        f" (main-lobe peak {pks[0]:+.2f} dB)" if len(pks) == 1 else
        " (main-lobe peaks " + ", ".join(f"{v:+.2f}" for v in pks) + " dB)")
    a1.plot(t, 20 * np.log10(a["rec"] / pk + 1e-12), color="tab:red", lw=0.9, zorder=2,
            label=f"reconstruction from the {cfg.Nrx} channels, no SATA" + note)
    a1.plot(t, 20 * np.log10(a["ref"] / pk + 1e-12), color="0.2", lw=0.9, zorder=3,
            label="reference: monostatic at the transmitter, full PRF")
    a1.set_xlim(t[0], t[-1]); a1.set_ylim(-100, 3)
    a1.set_xlabel("azimuth time [s]"); a1.set_ylabel("amplitude [dB re. reference peak]")
    a1.set_title("whole azimuth line", fontsize=10 if small else 11); a1.grid(alpha=0.3)
    h_, l_ = a1.get_legend_handles_labels(); a1.legend(h_[::-1], l_[::-1], fontsize=8.5 if small else 9.5, loc="upper center")


def draw_irf_zoom(a2, cfg, irf, t0=None, small=False):
    """Oversampled zoom on the main lobe that focuses at time t0 (default: the highest reference peak)."""
    tu, pk = irf["tu"], irf["pk"]
    t0 = irf["t"][irf["ipk"]] if t0 is None else t0
    near = np.abs(tu - t0) <= 0.012
    for k, sty in (("ref", dict(color="0.2", lw=1.8)), ("rec", dict(color="tab:red", lw=1.1, ls="--"))):
        a2.plot(1e3 * (tu[near] - t0), 20 * np.log10(irf["up"][k][near] / pk + 1e-12), **sty)
    a2.set_xlim(-8, 8); a2.set_ylim(-50, 3)
    a2.set_xlabel("azimuth time from the peak [ms]"); a2.set_title("main lobe", fontsize=10 if small else 11); a2.grid(alpha=0.3)


def draw_irf(a1, a2, cfg, irf, small=False):
    """a1: whole azimuth line; a2: zoom on the main lobe (oversampled)."""
    draw_irf_line(a1, cfg, irf, small)
    draw_irf_zoom(a2, cfg, irf, small=small)


def plot_irf(cfg, tr, x, h, png):
    irf = compute_irf(cfg, tr)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw=dict(width_ratios=[1.6, 1.0]))
    draw_irf(a1, a2, cfg, irf)
    fig.suptitle(f"IRF after reconstruction, channel 0 is the transmitter | one target at x = {x:.0f} m, h = {h:.0f} m | "
                 f"$N_{{rx}}$ = {cfg.Nrx}, PRF = {cfg.prf:.0f} Hz ({cfg.PRF_op:.0f} Hz per channel), $B_a$ = {cfg.abw:.0f} Hz\n"
                 f"$b_{{at}}$ = [" + ", ".join(f"{b:.2f}" for b in cfg.array.bat) + "] m, "
                 f"$b_{{xt}}$ = [" + ", ".join(f"{b:.2f}" for b in cfg.array.bxt) + "] m", fontsize=11.5)
    fig.text(0.5, 0.012, irf_note(cfg, irf), ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(png, dpi=130); plt.close(fig)
    return irf["peak"], irf["amb"], irf["err"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=float, default=0.0, help="target azimuth position [m]")
    ap.add_argument("--h", type=float, default=0.0, help="target height [m] (0 = on the reference point)")
    ap.add_argument("--nrx", type=int, default=4)
    ap.add_argument("--bxt", type=float, default=100.0, help="b_xt of channels 1.. ~ U(0, bxt) [m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tx_rx_channel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = S.make_config([(a.x, a.h)], nrx=a.nrx, bxt_max=a.bxt, seed=a.seed, tx_rx_channel0=True)
    tr = S.build_platform_tracks(cfg)
    print(f"b_at = {np.round(cfg.array.bat, 3)} m\nb_xt = {np.round(cfg.array.bxt, 3)} m")

    plot_geometry(cfg, os.path.join(a.out, "geometry.png"))
    rows = plot_range_fit(cfg, tr, os.path.join(a.out, "range_fit.png"))
    print(f"\n{'ch':>2} {'a0 [m]':>12} {'a1 [m/s]':>11} {'a2 [m/s^2]':>13} {'C0 [m]':>12} {'C1 [s]':>11} {'C2 [s^2/m]':>11} {'Dt [ms]':>8}")
    for i, a0, a1, a2, C0, C1, C2, Dt in rows:
        print(f"{i:2d} {a0:12.6f} {a1:11.3e} {a2:13.6e} {C0:12.6f} {C1:11.3e} {C2:11.3e} {1e3 * Dt:8.3f}")
    peak, amb, err = plot_irf(cfg, tr, a.x, a.h, os.path.join(a.out, "irf.png"))
    print(f"\nIRF, target x = {a.x:.0f} m, h = {a.h:.0f} m: reconstruction main-lobe peak {peak:+.3f} dB re. reference, "
          f"highest ambiguity {amb['rec']:.1f} dB (reference {amb['ref']:.1f} dB), "
          f"max |reconstruction - reference limited to B_a| / peak = {err:.2e}")
    print(f"wrote geometry.png, range_fit.png, irf.png in {a.out}/")


if __name__ == "__main__":
    main()
