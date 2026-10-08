"""
Figures that explain the SATA 1D kernel of the reference code (sata_for_topography in SATA1D.txt),
step by step. The kernel below follows the reference: sub-apertures of sqrt(wl r / 2), 50 % overlap,
zero padding, FFT, Doppler -> angle -> azimuth cell, phase -2 pi / wl * delta_C0[cell], IFFT, linear
cross-fade. Signals come from the sata1d signal model. Writes in --out:

    sata_3_blocks.png         the sub-apertures and their weights
    sata_4_time_doppler.png   time-Doppler plane of three targets; one sub-aperture spectrum as a low-resolution image
    sata_5_table_phase.png    the delta_C0 table and the phase the kernel reads for every (sub-aperture, bin)
    sata_6_effect.png         a position-dependent phase error removed by the kernel
    sata_7_folding.png        the same plane for one channel (sampled below the Doppler bandwidth)
    sata_1_pipeline.png          where the kernel sits in the reconstruction chain (block diagram)
    sata_8_in_reconstruction.png the kernel on every channel before the reconstruction: IRF and the tables
    sata_2_geometry_table.png where the delta_C0 table comes from: geometry of a cell, delta_C0 against height, heights along the line

    python sata_review_figures.py --out ../results/sata1d/figures/sata_review
"""
import argparse
import ast
import contextlib
import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sata1d as S

XS = (-2000.0, 0.0, 2000.0)            # azimuth positions of the three targets [m]
COL = ("tab:red", "tab:green", "tab:blue")
LAB = lambda x: "0" if x == 0 else f"{x:+.0f}"   # position label without "+0"
HALF = 1000.0                          # half-width of the region of the table that holds the elevated target [m]


def sizes(wl, rref, prf_data, v, osf):
    """Sub-aperture sizes of the reference kernel for data sampled at prf_data."""
    deltax = np.sqrt(wl * rref / 2)
    Tsubeff = int(np.round(deltax * prf_data / v * 0.5) * 2)
    Nzp = int(2 ** np.ceil(np.log2(Tsubeff)) * osf)
    return deltax, Tsubeff, Tsubeff // 2, Nzp


def ruler(wl, r, prf_data, v, Nzp):
    """Doppler of every FFT bin (FFT order), its angle and its azimuth offset in cells of the data."""
    f = np.fft.fftfreq(Nzp, 1.0 / prf_data)
    f[Nzp // 2] = prf_data / 2                                   # edge bin labelled +prf/2, as in the reference (fsub[0])
    beta = np.arcsin(wl * f / (2 * v))
    return f, beta, r * np.tan(beta) / v * prf_data


def sata_reference(data, dC0, wl, rref, prf_data, v, r, osf=1, inverse=False, apply=True):
    """The loop of the reference kernel. Returns the corrected line, the sub-aperture spectra (before the
    correction) and the phase read from the table, both [Nsub, Nzp], and the block centres [cells]."""
    dimx = len(data)
    _, Tsubeff, Tsub, Nzp = sizes(wl, rref, prf_data, v, osf)
    _, _, azpos = ruler(wl, r, prf_data, v, Nzp)
    up = np.arange(Tsub) / (Tsub - 1.0)
    w = np.r_[up, up[::-1]]                                      # cross-fade: rises over the first half, falls over the second
    out = np.zeros(dimx, complex); norm = np.zeros(dimx)
    spec, phase, centre = [], [], []
    for m in range(dimx // Tsub - 1):
        i0 = m * Tsub; posix = i0 + Tsub
        aux = np.fft.fft(data[i0:i0 + Tsubeff], Nzp)                # zero padding + FFT
        posaux = np.clip((azpos + posix).astype(int), 0, dimx - 1) # the cell every bin looks at
        ph = -2 * np.pi / wl * dC0[posaux]
        spec.append(aux); phase.append(ph); centre.append(posix)
        if apply:
            aux = aux * np.exp(-1j * ph if inverse else 1j * ph)    # as in the reference: direct exp(-j 2 pi dC0 / wl), inverse exp(+j 2 pi dC0 / wl)
        out[i0:i0 + Tsubeff] += w * np.fft.ifft(aux)[:Tsubeff]
        norm[i0:i0 + Tsubeff] += w
    ok = norm > 1e-9
    out[ok] /= norm[ok]; out[~ok] = data[~ok]
    return out, np.array(spec), np.array(phase), np.array(centre)


def setup():
    cfg = S.make_config([(x, 0.0) for x in XS], bxt_max=0.0, tx_rx_channel0=True)
    tr = S.build_platform_tracks(cfg)
    return cfg, tr


def one_target(cfg, tr, j):
    """Monostatic full-PRF signal of target j alone."""
    return S.generate_reference(cfg, tr, cfg.scene.points[1 + j][None, :])


def fig_blocks(cfg, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    deltax, Tsubeff, Tsub, Nzp = sizes(wl, r0, prf, v, 1)
    up = np.arange(Tsub) / (Tsub - 1.0); w = np.r_[up, up[::-1]]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    tot = np.zeros(6 * Tsub)
    cols = plt.get_cmap("tab10")(np.arange(5))
    for m in range(4):
        n = np.arange(m * Tsub, m * Tsub + Tsubeff)
        ax.fill_between(n, 0, w, color=cols[m], alpha=0.25, lw=0)
        ax.plot(n, w, color=cols[m], lw=1.6, label=f"sub-aperture {m}")
        ax.plot([m * Tsub + Tsub] * 2, [0, 1.0], color=cols[m], lw=0.8, ls=":")
        tot[n] += w
    ax.plot(np.arange(Tsub, 4 * Tsub), tot[Tsub:4 * Tsub], color="k", lw=1.8, ls="--", label="sum of the weights = 1")
    ax.annotate("", xy=(Tsubeff, 1.13), xytext=(0, 1.13), arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(1.5, 1.16, f"one sub-aperture: {Tsubeff} samples = {Tsubeff * v / prf:.0f} m of flight "
            f"(optimum $\\sqrt{{\\lambda r/2}}$ = {deltax:.0f} m)", ha="left", va="bottom", fontsize=10)
    ax.annotate("", xy=(2 * Tsub, -0.13), xytext=(Tsub, -0.13), arrowprops=dict(arrowstyle="<->", color="k"), annotation_clip=False)
    ax.text(1.5 * Tsub, -0.17, f"hop: {Tsub} samples (50 % overlap)", ha="center", va="top", fontsize=10)
    ax.set_xlim(0, 5 * Tsub); ax.set_ylim(0, 1.36)
    ax.set_xlabel("azimuth sample of the line (here at PRF = %.0f Hz)" % prf, labelpad=18); ax.set_ylabel("weight in the overlap-add")
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="center right", ncol=1)
    ax.set_title("The line is cut into overlapping sub-apertures; each is corrected on its own and they are cross-faded back", fontsize=10.5)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)


def fig_time_doppler(cfg, tr, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    sig = S.generate_reference(cfg, tr)
    osf = 2
    _, spec, _, centre = sata_reference(sig, np.zeros(cfg.Na), wl, r0, prf, v, r0, osf=osf, apply=False)
    deltax, Tsubeff, Tsub, Nzp = sizes(wl, r0, prf, v, osf)
    f, beta, azpos = ruler(wl, r0, prf, v, Nzp)
    o = np.argsort(f)
    t = (centre - cfg.Na / 2) / prf
    A = np.abs(spec[:, o]); A = 20 * np.log10(A / A.max() + 1e-6)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    im = a1.imshow(A.T, origin="lower", aspect="auto", extent=[t[0], t[-1], f[o][0], f[o][-1]], cmap="magma", vmin=-35, vmax=0)
    Ka = -2 * v ** 2 / (wl * r0)
    for x, c in zip(XS, COL):
        a1.plot(x / v, 0, "o", ms=7, mfc=c, mec="w", mew=1.0)
    a1.text(-2.95, -940, "dots: zero Doppler of each target,\nreached at t = x/v (targets at\nx = %s, %s, %s m, left to right)" % tuple(LAB(x) for x in XS), color="w", fontsize=9, va="bottom")
    m0 = int(np.argmin(np.abs(t)))
    a1.axvline(t[m0], color="w", lw=1.2, ls="--")
    a1.text(t[m0] + 0.04, 900, "one sub-aperture\n(right panel)", color="w", fontsize=9, va="top")
    a1.set_xlabel("time of the sub-aperture centre [s]"); a1.set_ylabel("Doppler frequency of the bin [Hz]")
    a1.set_title("Spectra of all sub-apertures: each target is a line of slope $-2v^2/\\lambda r$ = %.0f Hz/s" % Ka, fontsize=10.5)
    fig.colorbar(im, ax=a1, pad=0.01, label="[dB]")
    xg = azpos[o] * v / prf
    a2.plot(xg, A[m0], color="0.15", lw=1.3)
    for x, c in zip(XS, COL):
        a2.axvline(x, color=c, lw=1.2, ls="--")
        a2.text(x, 2.0, LAB(x) + " m", color=c, ha="center", fontsize=9)
    a2.set_xlim(-6000, 6000); a2.set_ylim(-40, 6); a2.grid(alpha=0.3)
    a2.set_xlabel("azimuth position that the ruler gives to each bin, $x = r\\,\\tan\\beta$ [m]")
    a2.set_ylabel("sub-aperture spectrum [dB]")
    a2.set_title("Spectrum of that sub-aperture = low-resolution image (%.0f m)" % deltax, fontsize=10.5)
    sec = a2.secondary_xaxis("top", functions=(lambda x: 2 * v / wl * np.sin(np.arctan(x / r0)), lambda fq: r0 * np.tan(np.arcsin(np.clip(wl * fq / (2 * v), -0.99, 0.99)))))
    sec.set_xlabel("Doppler frequency of the bin [Hz]", fontsize=9)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    pk = [xg[np.argmax(np.where(np.abs(xg - x) < 600, A[m0], -99))] for x in XS]
    print("time-Doppler: sub-aperture", Tsubeff, "samples, Nzp", Nzp, "| peaks of the central sub-aperture at x =", np.round(pk, 0), "m (targets", XS, ")")


def table_for(cfg, x_target, value, halfwidth_m, prf_data):
    """delta_C0 table on the grid of the data: `value` within +- halfwidth_m of the target, zero elsewhere."""
    n = int(round(len(cfg.ta) * prf_data / cfg.prf))
    xcell = (np.arange(n) - n / 2) * cfg.system.vs / prf_data
    return xcell, np.where(np.abs(xcell - x_target) <= halfwidth_m, value, 0.0)


def fig_table_phase(cfg, tr, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    deltax = np.sqrt(wl * r0 / 2)
    dval = wl / 4
    xcell, dC0 = table_for(cfg, XS[0], dval, HALF, prf)
    sig = S.generate_reference(cfg, tr)
    osf = 2
    _, spec, phase, centre = sata_reference(sig, dC0, wl, r0, prf, v, r0, osf=osf, apply=False)
    _, Tsubeff, Tsub, Nzp = sizes(wl, r0, prf, v, osf)
    f, _, _ = ruler(wl, r0, prf, v, Nzp); o = np.argsort(f)
    t = (centre - cfg.Na / 2) / prf
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw=dict(width_ratios=[0.85, 1.0, 1.0]))
    a0, a1, a2 = axs
    a0.plot(xcell / 1e3, 1e3 * dC0, color="tab:red", lw=2)
    for x, c in zip(XS, COL):
        a0.axvline(x / 1e3, color=c, lw=1.0, ls="--")
    a0.set_xlim(-8, 8); a0.set_xlabel("azimuth position of the cell [km]"); a0.set_ylabel("$\\Delta C_0$ stored in the table [mm]")
    a0.set_title("1) the table: one $\\Delta C_0$ per azimuth cell", fontsize=10.5); a0.grid(alpha=0.3)
    a0.text((XS[0] - HALF) / 1e3 - 0.3, 1e3 * dval, "elevated target\n($\\Delta C_0=\\lambda/4$)", color="tab:red", ha="right", va="center", fontsize=9)
    a0.text(XS[2] / 1e3, 4, "flat targets:\n$\\Delta C_0 = 0$", color="0.3", ha="center", fontsize=9)
    a0.set_ylim(-3, 1e3 * dval * 1.35)
    P = np.degrees(phase[:, o])
    im1 = a1.imshow(-P.T, origin="lower", aspect="auto", extent=[t[0], t[-1], f[o][0], f[o][-1]], cmap="Reds", vmin=0, vmax=90)
    a1.set_title("2) phase read from the table for every\n(sub-aperture, bin): $2\\pi\\,\\Delta C_0[\\,$cell$\\,]/\\lambda$", fontsize=10.5)
    fig.colorbar(im1, ax=a1, pad=0.01, label="correction phase [deg]")
    A = np.abs(spec[:, o]); A = 20 * np.log10(A / A.max() + 1e-6)
    a2.imshow(A.T, origin="lower", aspect="auto", extent=[t[0], t[-1], f[o][0], f[o][-1]], cmap="Greys", vmin=-35, vmax=0)
    a2.contour(t, f[o], (-P.T > 45).astype(float), levels=[0.5], colors="tab:red", linewidths=1.6)
    a2.set_title("3) the same region (red outline) over the signal:\nit covers the elevated target and only it", fontsize=10.5)
    for ax in (a1, a2):
        ax.set_xlabel("time of the sub-aperture centre [s]"); ax.set_ylabel("Doppler frequency of the bin [Hz]"); ax.set_xlim(-1.6, 1.6)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)


def fig_effect(cfg, tr, png):
    wl, r0, v, prf = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf
    deltax = np.sqrt(wl * r0 / 2)
    dval = wl / 4
    xcell, dC0 = table_for(cfg, XS[0], dval, HALF, prf)
    parts = [one_target(cfg, tr, j) for j in range(3)]
    bad = parts[0] * np.exp(-2j * np.pi * dval / wl) + parts[1] + parts[2]       # only the first target carries the error exp(-j 2 pi dC0 / wl); the inverse branch removes it
    fixed, _, _, _ = sata_reference(bad, dC0, wl, r0, prf, v, r0, osf=4, inverse=True)
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    F = lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * np.conj(np.fft.fft(sref))), cfg.Na // 2)
    t = (np.arange(cfg.Na) - cfg.Na // 2) / prf
    ideal, fb, ff = F(sum(parts)), F(bad), F(fixed)
    pk = np.abs(ideal).max()
    rows = []
    for x in XS:
        i = int(np.argmin(np.abs(t - x / v))); i = i - 3 + int(np.argmax(np.abs(ideal[i - 3:i + 4])))
        rows.append((x, np.degrees(np.angle(fb[i] / ideal[i])), np.degrees(np.angle(ff[i] / ideal[i])),
                     20 * np.log10(abs(fb[i]) / abs(ideal[i])), 20 * np.log10(abs(ff[i]) / abs(ideal[i]))))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    a1.plot(t, 20 * np.log10(np.abs(fb) / pk + 1e-9), color="tab:red", lw=1.0, label="before the kernel")
    a1.plot(t, 20 * np.log10(np.abs(ff) / pk + 1e-9), color="tab:blue", lw=1.0, ls="--", label="after the kernel")
    a1.set_xlim(-0.6, 0.6); a1.set_ylim(-60, 6); a1.grid(alpha=0.3); a1.legend(fontsize=9, loc="upper right")
    a1.set_xlabel("azimuth time [s]"); a1.set_ylabel("focused amplitude [dB]")
    a1.set_title("Focused line: the three peaks keep their amplitude (within %.2f dB)" % max(abs(r_[4] - r_[3]) for r_ in rows), fontsize=10.5)
    away = np.all([np.abs(t - x / v) > 0.03 for x in XS], axis=0) & (np.abs(t) < 0.9)
    spur = 20 * np.log10(np.abs(ff[away]).max() / pk)
    a1.text(-0.585, -7, "small spikes after the kernel\n(highest: %.0f dB): energy of\nthe elevated target that\nspreads past the edge of\nits region of the table" % spur, fontsize=8.5, va="top", color="tab:blue")
    xi = np.arange(3); wbar = 0.36
    a2.bar(xi - wbar / 2, [r_[1] for r_ in rows], wbar, color="tab:red", label="before the kernel")
    a2.bar(xi + wbar / 2, [r_[2] for r_ in rows], wbar, color="tab:blue", label="after the kernel")
    for k, r_ in enumerate(rows):
        a2.text(k - wbar / 2, r_[1] - 4 if r_[1] < -5 else r_[1] + 2, f"{round(r_[1], 1) + 0.0:+.1f}°", ha="center", va="top" if r_[1] < -5 else "bottom", fontsize=9)
        a2.text(k + wbar / 2, r_[2] + 2, f"{round(r_[2], 1) + 0.0:+.1f}°", ha="center", va="bottom", fontsize=9)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(xi); a2.set_xticklabels([f"x = {LAB(x)} m\n" + ("elevated ($\\Delta C_0=\\lambda/4$)" if k == 0 else "flat") for k, x in enumerate(XS)])
    a2.set_ylim(-110, 25); a2.set_ylabel("phase error of the target's peak [deg]"); a2.grid(alpha=0.3, axis="y"); a2.legend(fontsize=9, loc="lower right")
    a2.set_title("Phase of each peak: only the elevated target is rotated back", fontsize=10.5)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("effect (table half-width %.0f m): highest level away from the three peaks, before %.1f dB, after %.1f dB" % (
        HALF, 20 * np.log10(np.abs(fb[away]).max() / pk), 20 * np.log10(np.abs(ff[away]).max() / pk)))
    for hw in (deltax, 500.0, 1000.0, 1500.0):
        _, d2 = table_for(cfg, XS[0], dval, hw, prf)
        f2 = F(sata_reference(bad, d2, wl, r0, prf, v, r0, osf=4, inverse=True)[0])
        i = int(np.argmin(np.abs(t - XS[0] / v))); i = i - 3 + int(np.argmax(np.abs(ideal[i - 3:i + 4])))
        print("  half-width %5.0f m: elevated target %+.3f dB, %+.2f deg, highest level away from the peaks %.1f dB" % (
            hw, 20 * np.log10(abs(f2[i]) / abs(ideal[i])), np.degrees(np.angle(f2[i] / ideal[i])), 20 * np.log10(np.abs(f2[away]).max() / pk)))
    print("effect: target x, phase before, phase after [deg], amplitude before, after [dB]")
    for r_ in rows:
        print("  %+6.0f  %+7.2f  %+7.2f  %+6.3f  %+6.3f" % r_)


def fig_folding(cfg, tr, png):
    wl, r0, v, prf, prfc = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf, cfg.PRF_op
    ch = S.generate_channels(cfg, tr)[0]                              # channel 0 = the transmitter itself, sampled at PRF/Nrx
    osf = 4
    _, spec, _, centre = sata_reference(ch, np.zeros(cfg.Na_ch), wl, r0, prfc, v, r0, osf=osf, apply=False)
    deltax, Tsubeff, Tsub, Nzp = sizes(wl, r0, prfc, v, osf)
    f, _, azpos = ruler(wl, r0, prfc, v, Nzp); o = np.argsort(f)
    t = (centre - cfg.Na_ch / 2) / prfc
    A = np.abs(spec[:, o]); A = 20 * np.log10(A / A.max() + 1e-6)
    X = r0 * np.tan(np.arcsin(wl * prfc / (2 * v)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    im = a1.imshow(A.T, origin="lower", aspect="auto", extent=[t[0], t[-1], f[o][0], f[o][-1]], cmap="magma", vmin=-35, vmax=0)
    a1.set_xlim(-1.7, 1.7); a1.set_xlabel("time of the sub-aperture centre [s]"); a1.set_ylabel("Doppler frequency of the bin [Hz]")
    a1.set_title("One channel (sampled at %.0f Hz): the lines of the targets fold inside $\\pm$%.0f Hz" % (prfc, prfc / 2), fontsize=10.5)
    fig.colorbar(im, ax=a1, pad=0.01, label="[dB]")
    Ka = -2 * v ** 2 / (wl * r0); T = wl * r0 / (cfg.system.La * v)       # illumination time
    tt = np.linspace(-T / 2, T / 2, 2001)
    ftrue = Ka * tt
    fb = (ftrue + prfc / 2) % prfc - prfc / 2
    xas = v * tt + r0 * np.tan(np.arcsin(wl * fb / (2 * v)))
    a2.plot(tt, xas / 1e3, ".", ms=2.5, color="tab:red", label="position the ruler gives to the energy of the target")
    a2.axhline(0, color="0.2", lw=1.2, ls="--", label="true position of the target (x = 0)")
    for m_ in (-1, 1):
        a2.text(m_ * 0.72, m_ * (X / 1e3 + 1.2), f"{m_ * X / 1e3:+.2f} km", color="tab:red", ha="center", va="center", fontsize=9)
    a2.axvspan(-prfc / 2 / abs(Ka), prfc / 2 / abs(Ka), color="tab:green", alpha=0.12, lw=0)
    a2.text(0, -X / 1e3 * 2.1, "|Doppler| < %.0f Hz:\nthe ruler is right\n(%.0f %% of the energy)" % (prfc / 2, 100 * prfc / (abs(Ka) * T)), color="tab:green", ha="center", va="center", fontsize=9)
    a2.set_xlim(-T / 2 * 1.05, T / 2 * 1.05); a2.set_ylim(-2.7 * X / 1e3, 2.7 * X / 1e3); a2.grid(alpha=0.3)
    a2.set_xlabel("time relative to the closest approach of the target [s]"); a2.set_ylabel("azimuth position [km]")
    a2.set_title("Where the kernel looks in the table for the target at x = 0", fontsize=10.5); a2.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print(f"folding: channel sub-aperture {Tsubeff} samples, Nzp {Nzp}, fold distance X = {X:.1f} m = {X * prfc / v:.1f} cells, "
          f"energy with |f| < PRF_ch/2: {100 * prfc / (abs(Ka) * T):.1f} %")


BXT = (0.0, 100.0, 180.0, 300.0)       # cross-track baselines of the example table [m]; channel 0 is on the transmitter


def delta_c0(h, bxt, r0, y0):
    """Exact (cross-track plane) and closed-form delta_C0 of a point at height h on the iso-range r0 of the reference point."""
    H = np.sqrt(r0 ** 2 - y0 ** 2)
    y = np.sqrt(r0 ** 2 - (H - h) ** 2)
    exact = np.sqrt((y - bxt) ** 2 + (H - h) ** 2) - np.sqrt((y0 - bxt) ** 2 + H ** 2)
    return exact, -bxt * h / (r0 * y0 / H)


def fig_geometry_table(cfg, png):
    wl, r0, v = cfg.system.wl, cfg.scene.r0, cfg.system.vs
    y0 = float(cfg.scene.ptg[1])
    fig = plt.figure(figsize=(16.5, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 0.9, 1.0], height_ratios=[0.8, 1.0])
    a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[:, 1]); a2 = fig.add_subplot(gs[0, 2]); a3 = fig.add_subplot(gs[1, 2], sharex=a2)
    # 1) cross-track plane, drawn out of scale
    Hd, yd, bd, hd = 10.0, 6.0, 1.7, 2.3
    rd = np.hypot(Hd, yd); ypd = np.sqrt(rd ** 2 - (Hd - hd) ** 2)
    ang = np.linspace(np.arctan2(yd - 1.6, Hd + 0.9), np.arctan2(ypd + 1.3, Hd - hd - 1.6), 80)
    a0.plot(rd * np.sin(ang), Hd - rd * np.cos(ang), color="0.55", lw=1.0, ls=":")
    a0.text(rd * np.sin(ang[-1]) + 0.1, Hd - rd * np.cos(ang[-1]) + 0.25, "same range $r_0$\nfrom the transmitter", color="0.4", fontsize=8.5, ha="center", va="bottom")
    yy = np.linspace(-1.2, 11.5, 300)
    a0.fill_between(yy, -0.6, hd * np.exp(-((yy - ypd) / 1.1) ** 2), color="0.88", lw=0)
    a0.plot(yy, hd * np.exp(-((yy - ypd) / 1.1) ** 2), color="0.5", lw=1.0)
    a0.plot([0, yd], [Hd, 0], color="tab:blue", lw=1.5); a0.plot([bd, yd], [Hd, 0], color="tab:blue", lw=1.5, ls="--")
    a0.plot([0, ypd], [Hd, hd], color="tab:red", lw=1.5); a0.plot([bd, ypd], [Hd, hd], color="tab:red", lw=1.5, ls="--")
    a0.plot([0, 0], [Hd, 0], color="0.5", lw=0.8, ls=":")
    a0.plot(0, Hd, "^", ms=11, color="k"); a0.plot(bd, Hd, "s", ms=8, color="k")
    a0.plot(yd, 0, "o", ms=8, color="tab:blue"); a0.plot(ypd, hd, "o", ms=8, color="tab:red")
    a0.text(-0.15, Hd + 0.35, "transmitter", ha="center", va="bottom", fontsize=9)
    a0.text(bd + 0.1, Hd + 0.35, "receiver $i$", ha="left", va="bottom", fontsize=9)
    a0.annotate("", xy=(bd, Hd - 0.55), xytext=(0, Hd - 0.55), arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
    a0.text(bd + 0.75, Hd - 0.5, "$b_{xt,i}$", ha="left", va="center", fontsize=9)
    a0.text(-0.2, Hd / 2, "$H$", ha="right", va="center", fontsize=9, color="0.4")
    th = np.linspace(0, np.arctan2(yd, Hd), 30)
    a0.plot(2.2 * np.sin(th), Hd - 2.2 * np.cos(th), color="0.4", lw=0.8); a0.text(0.55, Hd - 2.75, "$\\theta$", fontsize=10, color="0.3")
    a0.text(yd - 0.2, -0.25, "reference point\n($h = 0$)", color="tab:blue", ha="right", va="top", fontsize=9)
    a0.text(ypd + 0.3, hd + 0.1, "target at\nheight $h$", color="tab:red", ha="left", va="bottom", fontsize=9)
    a0.plot([], [], color="0.2", lw=1.5, label="range to the transmitter, $R_T$"); a0.plot([], [], color="0.2", lw=1.5, ls="--", label="range to receiver $i$, $R_{R,i}$")
    a0.legend(fontsize=8.5, loc="upper right", frameon=False)
    a0.set_xlim(-1.4, 11.6); a0.set_ylim(-1.9, 11.6); a0.set_aspect("equal"); a0.set_xticks([]); a0.set_yticks([])
    a0.set_xlabel("ground range (cross-track)   -   drawing not to scale"); a0.set_ylabel("height")
    a0.set_title("1) geometry of one azimuth cell:\n$C_{0,i} = R_{R,i} - R_T$,   $\\Delta C_{0,i} = C_{0,i}$(target) $-\\ C_{0,i}$(reference point)", fontsize=10.5)
    # 2) delta_C0 against the height
    h = np.linspace(0, 1000, 201)
    cols = ("0.3", "tab:orange", "tab:purple", "tab:brown")
    for i, b in enumerate(BXT):
        ex, cf = delta_c0(h, b, r0, y0)
        a1.plot(h, ex, color=cols[i], lw=1.8, label="channel %d, $b_{xt}$ = %.0f m" % (i, b))
        a1.plot(h[::25], cf[::25], "o", ms=4, mfc="none", color=cols[i])
    a1.plot([], [], "o", ms=4, mfc="none", color="0.3", label="closed form $-b_{xt}\\,h/(r_0\\tan\\theta)$")
    a1.set_xlabel("height of the cell above the reference point, $h$ [m]"); a1.set_ylabel("$\\Delta C_{0,i}$ [m]"); a1.grid(alpha=0.3); a1.legend(fontsize=8.5, loc="lower left")
    sec = a1.secondary_yaxis("right", functions=(lambda d: d / wl, lambda c: c * wl)); sec.set_ylabel("phase $2\\pi\\Delta C_0/\\lambda$ [cycles]")
    a1.set_title("2) the height of the cell gives its $\\Delta C_0$\n(one curve per channel)", fontsize=10.5)
    # 3) a terrain profile along the line becomes the table
    x = (np.arange(cfg.Na) - cfg.Na / 2) * v / cfg.prf
    sx = np.clip((x + 2600.0) / 5200.0, 0, 1); hx = 1000.0 * (3 * sx ** 2 - 2 * sx ** 3)
    a2.fill_between(x / 1e3, 0, hx, color="0.88", lw=0); a2.plot(x / 1e3, hx, color="0.4", lw=1.3)
    for xt in (-2600.0, 0.0, 2600.0):
        ht = float(np.interp(xt, x, hx))
        a2.plot(xt / 1e3, ht, "o", ms=6, color="tab:red")
        a2.text(xt / 1e3, ht + 110, "%.0f m" % ht, color="tab:red", ha="center", fontsize=8.5)
    a2.set_ylim(-60, 1450); a2.set_ylabel("height $h$ [m]"); a2.grid(alpha=0.3); plt.setp(a2.get_xticklabels(), visible=False)
    a2.set_title("3) heights along the line (DEM; dots: point targets)\nbecome the table, one $\\Delta C_0$ per azimuth cell and per channel", fontsize=10.5)
    for i, b in enumerate(BXT):
        a3.plot(x / 1e3, delta_c0(hx, b, r0, y0)[0], color=cols[i], lw=1.8, label="channel %d" % i)
    a3.set_xlim(-6, 6); a3.set_xlabel("azimuth position of the cell [km]"); a3.set_ylabel("table $\\Delta C_{0,i}$[cell] [m]"); a3.grid(alpha=0.3); a3.legend(fontsize=8.5, loc="lower left", ncol=2)
    fig.tight_layout(); fig.savefig(png, dpi=170); plt.close(fig)
    print("geometry/table: delta_C0 at h = 1000 m, exact and closed form [m]:", [tuple(np.round([float(q) for q in delta_c0(1000.0, b, r0, y0)], 4)) for b in BXT])


REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "sata", "reference_code_SATA1D.py")
OSF = 4


def load_reference_kernel(path=REF_FILE):
    """sata_for_topography exactly as written in the reference file (only that function is executed; its prints are muted)."""
    src = open(path, encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "sata_for_topography")
    ns = {"np": np, "plt": plt}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    fn = ns["sata_for_topography"]

    def quiet(*a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)
    return quiet


def fig_pipeline(png):
    """Block diagram: where the SATA kernel sits in the multichannel reconstruction."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(16, 5.9)); ax.set_xlim(0, 16); ax.set_ylim(0, 5.9); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.5, ha="center"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.5))
        ax.text(x + (w / 2 if ha == "center" else 0.15), y + h / 2, text, ha=ha, va="center", fontsize=fs, linespacing=1.35)

    def arrow(x0, y0, x1, y1, c="0.25"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color=c))

    G, B, O, K = ("#e9f5ec", "#2e8b57"), ("#e8f0fb", "#1f5fa8"), ("#fdf0e3", "#d9793a"), ("#f1f1f1", "#555555")
    box(0.2, 4.0, 3.0, 1.5, "heights along the azimuth line\n(DEM; in a simulation, the\nheights of the point targets)", *G)
    box(3.9, 4.0, 4.9, 1.5, "geometry of every azimuth cell:\ntransmitter, receiver $i$, reference point\n"
        "$\\Delta C_{0,i} = C_{0,i}$(terrain) $-\\ C_{0,i}$(reference point)", *G)
    arrow(3.25, 4.75, 3.85, 4.75, G[1])
    arrow(6.35, 3.95, 6.35, 3.25, G[1])
    ax.text(6.5, 3.6, "table $\\Delta C_{0,i}$[cell]: one value per azimuth cell", fontsize=9.5, color=G[1], va="center")
    box(0.2, 1.1, 2.4, 1.7, "channel $i$\nazimuth line sampled\nat PRF/$N$\n(sum of all targets)", *K)
    box(3.3, 0.6, 5.4, 2.6, "SATA kernel   (reference: sata_for_topography)\n"
        "1. cut the line into short sub-apertures\n2. FFT of each sub-aperture\n"
        "3. each Doppler bin $\\rightarrow$ the azimuth cell it looks at\n"
        "4. multiply the bin by $e^{\\,j 2\\pi\\,\\Delta C_{0,i}[\\mathrm{cell}]/\\lambda}$\n"
        "5. inverse FFT, cross-fade the sub-apertures", *B, ha="left")
    box(9.4, 1.1, 2.3, 1.7, "corrected channel $i$\n(as if every target were\nat the reference height)", *K)
    box(12.4, 0.6, 3.4, 2.6, "reconstruction (unchanged)\nFFT of the $N$ channels,\nfilters $P = N\\,H^{-1}$ of the\nreference point, combination,\ninverse FFT\n$\\rightarrow$ line at the full PRF", *O)
    arrow(2.65, 1.95, 3.25, 1.95); arrow(8.75, 1.95, 9.35, 1.95); arrow(11.75, 1.95, 12.35, 1.95)
    ax.text(8.0, 0.15, "The kernel runs once per channel, each with its own table (the channel on the transmitter has "
            "$\\Delta C_0 = 0$ and is left as it is). The reconstruction itself is not modified.", ha="center", va="center", fontsize=9.5, color="0.3")
    fig.savefig(png, dpi=170, bbox_inches="tight"); plt.close(fig)


def channel_tables(cfg, dC0, half_cells, copies):
    """One delta_C0 table per channel on the channel grid (cells of v / PRF_op): the value of each target within
    +- half_cells of the cell where the channel sees it; copies=True writes it also at the folded-Doppler positions."""
    n, v, prfc = cfg.Na_ch, cfg.system.vs, cfg.PRF_op
    X = cfg.scene.r0 * np.tan(np.arcsin(cfg.system.wl * prfc / (2 * v))) / v * prfc         # distance of one fold [cells]
    M = int(np.ceil((v / cfg.system.La + prfc / 2) / prfc)) - 1 if copies else 0           # folds that carry signal
    tabs = np.zeros((cfg.Nrx, n))
    for i in range(cfg.Nrx):
        for (dx, _, _), d in zip(cfg.scene.extra_offsets, dC0[:, i]):
            c = n / 2 + (cfg.scene.x0 + dx + cfg.array.bat[i] / 2) / v * prfc
            for m in range(-M, M + 1):
                lo, hi = int(round(c + m * X - half_cells)), int(round(c + m * X + half_cells))
                tabs[i, max(lo, 0):min(hi + 1, n)] = d
    return tabs, X


def fig_in_reconstruction(png, half_cells=60):
    """SATA on every channel before the reconstruction: two targets, one elevated."""
    from tx_rx_channel_cases import make_case
    from run_one_case import ambiguity_level
    targets = [(-1500.0, 1000.0), (1500.0, 0.0)]
    bxt = [0.0, 10.0, 18.0, 30.0]
    cfg = make_case(targets, bxt); tr = S.build_platform_tracks(cfg)
    wl, r0, v, prf, prfc, N = cfg.system.wl, cfg.scene.r0, cfg.system.vs, cfg.prf, cfg.PRF_op, cfg.Nrx
    dC0 = np.asarray(S.delta_C0_per_target(cfg, tr))                                       # [target, channel]
    ch = S.generate_channels(cfg, tr)
    try:
        kern = load_reference_kernel()
    except Exception as e:                                                                 # reference file not found
        print("reference file not used:", e); kern = None

    def sata(line, tab):
        if kern is None:
            return sata_reference(line, tab, wl, r0, prfc, v, r0, osf=OSF, inverse=True)[0]
        return kern(line.copy(), tab, r0, prfc, 1, v, 0.0, wl, r0, inverse=True, sata_osf=OSF)

    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    F = lambda s_: np.abs(np.roll(np.fft.ifft(np.fft.fft(s_) * np.conj(np.fft.fft(sref))), cfg.Na // 2))
    t = (np.arange(cfg.Na) - cfg.Na // 2) / prf
    dt = prfc * wl * r0 / (2 * v ** 2)
    xs = np.array([x for x, _ in targets])
    env = {"ref": F(S.generate_reference(cfg, tr)), "none": F(S.reconstruct(cfg, tr, ch))}
    pk = env["ref"].max()

    def run(half, copies):
        tabs, X = channel_tables(cfg, dC0, half, copies)
        return F(S.reconstruct(cfg, tr, np.array([sata(ch[i], tabs[i]) for i in range(N)]))), tabs, X

    def numbers(e):
        pks = []
        for x in xs:
            i = int(np.argmin(np.abs(t - x / v)))
            pks.append(20 * np.log10(e[i - 4:i + 5].max() / env["ref"][i - 4:i + 5].max()))
        return pks, ambiguity_level(t, e, pk, xs, v, dt)

    print("in the reconstruction: delta_C0 of the elevated target per channel [mm]:", np.round(1e3 * dC0[0], 2), "| in cycles:", np.round(dC0[0] / wl, 3))
    if kern is not None:
        tabs, _ = channel_tables(cfg, dC0, half_cells, True)
        a_, b_ = kern(ch[3].copy(), tabs[3], r0, prfc, 1, v, 0.0, wl, r0, inverse=True, sata_osf=OSF), sata_reference(ch[3], tabs[3], wl, r0, prfc, v, r0, osf=OSF, inverse=True)[0]
        print("  reference function as written vs the loop of this file, channel 3: max |difference| / max |signal| = %.1e" % (np.abs(a_ - b_).max() / np.abs(b_).max()))
    print("  %-28s main-lobe peaks [dB]: %s | highest ambiguity %.1f dB" % ("reference", "  0.00,  0.00", numbers(env["ref"])[1]))
    print("  %-28s main-lobe peaks [dB]: %s | highest ambiguity %.1f dB" % ("no SATA", ", ".join("%+.2f" % q for q in numbers(env["none"])[0]), numbers(env["none"])[1]))
    for half in (30, 60, 90):
        for copies in (False, True):
            e, _, _ = run(half, copies); pks, amb = numbers(e)
            print("  half-width %2d cells, %-12s main-lobe peaks [dB]: %s | highest ambiguity %.1f dB" % (half, "with copies" if copies else "targets only", ", ".join("%+.2f" % q for q in pks), amb))
    env["direct"], tab_d, X = run(half_cells, False)
    env["copies"], tab_c, _ = run(half_cells, True)
    res = {k: numbers(e) for k, e in env.items()}

    fig = plt.figure(figsize=(16.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.45, wspace=0.16)
    a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    db = lambda e: 20 * np.log10(e / pk + 1e-12)
    lab = {"none": "no SATA", "direct": "SATA whole band, table: targets only", "copies": "SATA whole band, table: targets and their folded-Doppler copies"}
    sty = {"none": dict(color="tab:red", lw=1.0), "direct": dict(color="tab:orange", lw=1.0), "copies": dict(color="tab:blue", lw=1.0)}
    for k in ("none", "direct", "copies"):
        a0.plot(t, db(env[k]), label="%s  (highest ambiguity %.1f dB)" % (lab[k], res[k][1]), **sty[k])
    a0.plot(t, db(env["ref"]), color="0.15", lw=0.9, label="reference: monostatic at the transmitter, full PRF  (%.1f dB)" % res["ref"][1])
    for x, (_, h), ha in zip(xs, targets, ("right", "left")):
        a0.text(x / v + (-0.04 if ha == "right" else 0.04), 3.0, "x = %+.0f m\nh = %.0f m" % (x, h), ha=ha, va="bottom", fontsize=8.5)
    a0.set_xlim(-2.6, 2.6); a0.set_ylim(-80, 16); a0.grid(alpha=0.3)
    a0.set_xlabel("azimuth time [s]"); a0.set_ylabel("amplitude [dB re. reference peak]")
    a0.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
    a0.set_title("IRF after reconstruction, $b_{xt}$ = [" + ", ".join("%g" % b for b in bxt) + "] m; the kernel runs on every channel before the reconstruction", fontsize=10.5)
    xc = (np.arange(cfg.Na_ch) - cfg.Na_ch / 2) * v / prfc / 1e3
    for ax, tab, ttl in ((a1, tab_d, "table of channel 3: targets only"),
                         (a2, tab_c, "table of channel 3: targets and the copies of the elevated one at $\\pm$%.2f km" % (X * v / prfc / 1e3))):
        ax.plot(xc, 1e3 * tab[3], color="tab:purple", lw=1.8)
        for x, c in zip(xs, ("tab:red", "0.3")):
            ax.axvline(x / 1e3, color=c, lw=1.0, ls="--")
        ax.set_ylabel("$\\Delta C_{0,3}$[cell] [mm]"); ax.grid(alpha=0.3); ax.set_title(ttl, fontsize=10.5); ax.set_xlim(-12, 12)
    a1.text(xs[0] / 1e3 - 1.4, 1e3 * dC0[0, 3] * 0.5, "elevated target", color="tab:red", fontsize=8.5, va="center", ha="right")
    a1.text(xs[1] / 1e3 + 0.4, 1e3 * dC0[0, 3] * 0.5, "target at the\nreference height", color="0.3", fontsize=8.5, va="center", ha="left")
    a2.set_xlabel("azimuth position of the cell [km]")
    fig.savefig(png, dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sata_review")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg, tr = setup()
    fig_pipeline(os.path.join(a.out, "sata_1_pipeline.png"))
    fig_geometry_table(cfg, os.path.join(a.out, "sata_2_geometry_table.png"))
    fig_blocks(cfg, os.path.join(a.out, "sata_3_blocks.png"))
    fig_time_doppler(cfg, tr, os.path.join(a.out, "sata_4_time_doppler.png"))
    fig_table_phase(cfg, tr, os.path.join(a.out, "sata_5_table_phase.png"))
    fig_effect(cfg, tr, os.path.join(a.out, "sata_6_effect.png"))
    fig_folding(cfg, tr, os.path.join(a.out, "sata_7_folding.png"))
    fig_in_reconstruction(os.path.join(a.out, "sata_8_in_reconstruction.png"))
    print("wrote 8 figures in", a.out)


if __name__ == "__main__":
    main()
