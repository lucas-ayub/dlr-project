# -*- coding: utf-8 -*-
"""
IRF 2-D de um alvo pontual focado -- contorno em range x azimute mais os dois
cortes 1-D, no estilo da Fig. 2.9 de Sakar.

Dois pontos que fazem esta figura funcionar (e sem os quais ela nao faz
sentido):

1.  FOCO POR FILTRO CASADO 2-D.  A imagem focada e'

        img = IFFT2( FFT2(x) * conj(FFT2(ref)) )

    com ``ref`` o sinal monostatico ideal do mesmo alvo.  Isso e' exato: a
    migracao de celula de range esta contida em ``ref``, portanto e'
    compensada sem nenhum RCMC aproximado.  Focar cada range bin
    separadamente com o chirp de azimute de um unico bin -- o atalho obvio --
    deixa a RCM (~41 m, varias celulas) sem corrigir e produz contornos
    tortos.

2.  RESOLUCAO EM RANGE COMPARAVEL A DE AZIMUTE.  No preset padrao
    ``res_rg = 30 m`` contra ``La/2 = 6 m`` em azimute: razao 5:1, e a cruz
    sai achatada.  O default aqui e' ``--res-rg 6``, que iguala as duas e
    reproduz a cruz isotropica da figura de referencia.  ``--res-rg 30``
    mostra o caso do preset padrao, com os eixos ajustados sozinhos.

Rodar de ``sar_reconstruction/``::

    python -m sata2d.run_irf2d                      # referencia monostatica
    python -m sata2d.run_irf2d --method sub         # SATA por sub-banda
    python -m sata2d.run_irf2d --res-rg 30          # preset padrao (anisotropico)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg))
    __package__ = os.path.basename(_pkg)

from .geometry import (make_params3d, build_tracks_3d, generate_reference_3d,
                       generate_channels_3d, CoeffTable3D)
from .reconstruction import (range_compress, build_delta_C0_map_3d,
                             reconstruct_subband_2d, scatterer_range,
                             range_bin_of, METHOD_KW)

C_LIGHT = 3.0e8
PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

LABEL = {"mono": "referência monostática", "no": "sem SATA",
         "whole": "SATA banda inteira", "sub": "SATA por sub-banda"}


# ---------------------------------------------------------------------------
def focus2d(x, ref):
    """Filtro casado 2-D contra o sinal monostatico ideal.

    Devolve a imagem focada com o alvo no centro do array (fftshift aplicado
    nos dois eixos).  Para ``x is ref`` isto e' a autocorrelacao 2-D, ou seja
    a IRF ideal do sistema.
    """
    R = np.fft.fft2(ref)
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x) * np.conj(R)))


def zoom2d(img, ia, ir, na, nr, zpa, zpr):
    """Zoom por zero-padding no dominio da frequencia, em torno de (ia, ir).

    Interpola o continuo entre as amostras -- necessario para um contorno
    liso.  A janela (2*na, 2*nr) tem que ser larga o bastante para conter os
    lobulos laterais relevantes, senao o wraparound da FFT contamina a borda.
    """
    w = img[ia - na:ia + na, ir - nr:ir + nr]
    W = np.fft.fftshift(np.fft.fft2(w))
    P = np.zeros((2 * na * zpa, 2 * nr * zpr), complex)
    P[na * zpa - na:na * zpa + na, nr * zpr - nr:nr * zpr + nr] = W
    # ifft2 divide pelo N MAIOR (ja' com zero-padding); reescala para manter
    # a amplitude da imagem original -- sem isto tudo desce 20log10(zpa*zpr) dB
    return np.fft.ifft2(np.fft.ifftshift(P)) * (zpa * zpr)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="mono", choices=("mono", "no", "whole", "sub"),
                    help="o que focar (default: mono)")
    ap.add_argument("--res-rg", type=float, default=6.0, dest="res_rg",
                    help="resolução em range [m] (default 6, iguala a de azimute)")
    ap.add_argument("--swath", type=float, default=300.0,
                    help="largura do swath [m] — controla Nr e o custo (default 300)")
    ap.add_argument("--azimuth", type=float, default=0.0, help="azimute do alvo [m]")
    ap.add_argument("--height", type=float, default=240.0, help="altura do alvo [m]")
    ap.add_argument("--nrx", type=int, default=4)
    ap.add_argument("--dxt", type=float, default=150.0)
    ap.add_argument("--sata-osf", type=int, default=4, dest="sata_osf")
    ap.add_argument("--span", type=float, default=None,
                    help="meio-tamanho dos eixos [m] (default: 5 células de resolução)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    specs = ((args.azimuth, args.height),) if (args.azimuth, args.height) != (0.0, 0.0) else ()
    p = make_params3d(Nrx=args.nrx, dxt=args.dxt, specs=specs,
                      res_rg=args.res_rg, swath=args.swath)
    tr = build_tracks_3d(p)
    ptg = np.asarray(p.points[0], float)
    nb = range_bin_of(p, scatterer_range(p, ptg))

    rho_r = C_LIGHT / (2.0 * p.rsf)
    d_az = p.ve / p.prf
    res_az = p.La / 2.0
    rcm = (p.ve * p.int_time / 2.0) ** 2 / (2.0 * p.r0)

    print(p.summary())
    print(f"resolução  : range {C_LIGHT/2/p.rbw:.2f} m | azimute {res_az:.2f} m")
    print(f"amostragem : range {rho_r:.2f} m | azimute {d_az:.2f} m")
    print(f"RCM        : {rcm:.1f} m = {rcm/rho_r:.1f} células de range")
    print(f"alvo       : azimute {args.azimuth:.0f} m, altura {args.height:.0f} m "
          f"-> range bin {nb}/{p.Nr}\n")

    ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)

    imgs = {"mono": focus2d(ref, ref)}
    if args.method != "mono":
        ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)
        tab = CoeffTable3D(p, tr, n_nodes=16)
        # a RCM tem que caber na largura do mapa de dC0, senão a correção
        # não acompanha o alvo ao longo da abertura
        hw = int(np.ceil(rcm / (2.0 * rho_r))) + 1
        maps = [build_delta_C0_map_3d(p, tr, i, range_halfwidth=hw)
                for i in range(p.Nrx)]
        print(f"  (range_halfwidth = {hw} células, para cobrir a RCM)")
        rec = reconstruct_subband_2d(p, tr, ch, tab, sata_osf=args.sata_osf,
                                     maps=maps, **METHOD_KW[args.method])
        imgs[args.method] = focus2d(rec, ref)

    key = args.method
    ia = int(np.argmax(np.abs(imgs["mono"]).max(axis=1)))
    ir = int(np.argmax(np.abs(imgs["mono"]).max(axis=0)))
    nrm = np.abs(imgs["mono"]).max()

    span = args.span if args.span is not None else 4.5 * max(res_az, C_LIGHT / 2 / p.rbw)
    na = max(8, int(np.ceil(3.0 * span / d_az)))
    nr = max(8, int(np.ceil(3.0 * span / rho_r)))
    zpa = max(1, int(np.ceil(24.0 * d_az / span)))
    zpr = max(1, int(np.ceil(24.0 * rho_r / span)))

    z = {k: zoom2d(v, ia, ir, na, nr, zpa, zpr) for k, v in imgs.items()}
    x_az = (np.arange(2 * na * zpa) - na * zpa) * d_az / zpa
    x_rg = (np.arange(2 * nr * zpr) - nr * zpr) * rho_r / zpr
    ca, cr = len(x_az) // 2, len(x_rg) // 2

    db = lambda v: 20.0 * np.log10(np.abs(v) / nrm + 1e-20)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(12.5, 5.6), dpi=160)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    lv = [-30, -25, -20, -16, -13, -10, -6, -3]
    ax0.contour(x_rg, x_az, db(z[key]), levels=lv, cmap="Spectral_r",
                linewidths=1.1, vmin=-34, vmax=0)
    ax0.set_xlabel("Range [m]"); ax0.set_ylabel("Azimuth [m]")
    ax0.set_xlim(-span, span); ax0.set_ylim(-span, span)
    ax0.set_aspect("equal"); ax0.grid(alpha=.3)
    ax0.set_title(f"IRF 2-D — {LABEL[key]}")

    for ax, cut, xax, lab in ((ax1, z[key][:, cr], x_az, "Azimuth [m]"),
                              (ax2, z[key][ca, :], x_rg, "Range [m]")):
        ax.plot(xax, db(cut), "k", lw=1.1)
        if key != "mono":
            ref_cut = z["mono"][:, cr] if lab.startswith("Azimuth") else z["mono"][ca, :]
            ax.plot(xax, db(ref_cut), color="0.6", lw=0.9, ls="--", label="mono")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel(lab); ax.set_ylabel("Impulse Response [dB]")
        ax.set_xlim(-2 * span, 2 * span); ax.set_ylim(-70, 3); ax.grid(alpha=.3)

    fig.suptitle(f"$N_{{rx}}$={p.Nrx} | PRF={p.prf:.0f} Hz | "
                 f"$B_a$={p.abw:.0f} Hz | $B_r$={p.rbw/1e6:.1f} MHz | "
                 f"$\\delta_r$={C_LIGHT/2/p.rbw:.1f} m, $\\delta_x$={res_az:.1f} m | "
                 f"$\\delta h$={args.height:.0f} m, $b_{{xt}}^{{max}}$={np.max(np.abs(p.bxt)):.0f} m",
                 fontsize="medium")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.11)

    out = args.out or os.path.join(PLOTS, f"irf2d_{key}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # métricas dos cortes
    for lab, cut, xax in (("azimute", z[key][:, cr], x_az), ("range", z[key][ca, :], x_rg)):
        a = np.abs(cut); a /= a.max()
        i0 = int(np.argmax(a))
        half = np.where(a >= 10 ** (-3 / 20))[0]
        irw = (xax[half[-1]] - xax[half[0]]) if half.size > 1 else np.nan
        mask = np.ones_like(a, bool)
        lo = hi = i0
        while lo > 0 and a[lo - 1] < a[lo]: lo -= 1
        while hi < len(a) - 1 and a[hi + 1] < a[hi]: hi += 1
        mask[lo:hi + 1] = False
        pslr = 20 * np.log10(a[mask].max()) if mask.any() else np.nan
        print(f"  {lab:>8}: IRW(-3dB) = {irw:6.2f} m | PSLR = {pslr:6.2f} dB")
    print(f"\nfigura escrita em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
