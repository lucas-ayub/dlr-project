"""Números da seção 'K e K_i na reconstrução padrão' do guia P1-P3 (v3). Só lê o pacote sata1d; não altera nada.
Exemplo dos guias: 5 alvos em x = -5200..5200 m, bxt = [0; 134,9; 20,5; 8,3] m, canal 0 no Tx."""
import json, os, sys
from dataclasses import replace
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import sata1d as S
from sata1d.config import ArrayGeometry
from sata1d.signal_model import raw
from sata1d.joint import reference_coefficients
from sata1d.joint_line import maps_from_heights

BXT = np.array([0.0, 134.9, 20.5, 8.3])
XS = np.array([-5200.0, -2600.0, 0.0, 2600.0, 5200.0])
mont = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
plano = lambda x: 0.0 * np.asarray(x, float)


def setup(xs, terrain):
    cfg = S.make_config(list(zip(xs, terrain(xs))), nrx=4, bxt_max=0.0, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=BXT))
    return cfg, S.build_platform_tracks(cfg)


def dB(e, r):
    n = np.linalg.norm(e) / np.linalg.norm(r)
    return None if n == 0 else round(float(20 * np.log10(n)), 1)


def filtros(cfg, tr):
    f = np.fft.fftfreq(cfg.Na, 1 / cfg.prf); wl = cfg.system.wl
    K = np.fft.fft(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]))
    H = np.array([np.exp(-2j * np.pi * (c[0] / wl + (-c[1] + c[3]) * f + c[2] * wl * f ** 2))
                  for c in reference_coefficients(cfg, tr)])
    return f, K, H


out = {}
# (a) eco de um ponto no canal i: K_i contra K H_i
cfg, tr = setup(XS, plano)
f, K, H = filtros(cfg, tr); band = np.abs(f) < cfg.abw / 2
ptg = cfg.scene.ptg[None, :]
Ki = np.array([np.fft.fft(raw(cfg, tr, ptg, tr.prx[i], tr.vrx[i])) for i in range(4)])
out["Ki_vs_KHi"] = [dB((Ki[i] - K * H[i])[band], Ki[i][band]) for i in range(4)]
out["Ki_vs_K"] = [dB((Ki[i] - K)[band], Ki[i][band]) for i in range(4)]
out["abw"] = cfg.abw; out["d"] = cfg.system.vs / cfg.prf


def cena(terrain):
    cfg, tr = setup(XS, terrain)
    f, K, H = filtros(cfg, tr); band = np.abs(f) < cfg.abw / 2
    U = np.fft.fft(S.generate_reference(cfg, tr))                       # monostático da cena, PRF cheia
    C = np.array([np.fft.fft(raw(cfg, tr, cfg.scene.points[1:], tr.prx[i], tr.vrx[i])) for i in range(4)])
    xp = cfg.scene.points[1:, 0] - cfg.scene.ptg[0]; hp = cfg.scene.points[1:, 2]
    dC0 = maps_from_heights(cfg, tr, hp)                                # [4, n_alvos]
    rampa = np.exp(-2j * np.pi * np.outer(f, xp / cfg.system.vs))       # [Na, n_alvos]
    G = rampa.sum(1)                                                    # cena focada, sem relevo
    GPsi = [(rampa * np.exp(-2j * np.pi * dC0[i] / cfg.system.wl)).sum(1) for i in range(4)]
    Urec = np.fft.fft(S.reconstruct(cfg, tr, S.generate_channels(cfg, tr)))
    return dict(
        U_vs_KG=dB((U - K * G)[band], U[band]),
        Ci_vs_HiU=[dB((C[i] - H[i] * U)[band], C[i][band]) for i in range(4)],
        Ci_vs_HiKGPsi=[dB((C[i] - H[i] * K * GPsi[i])[band], C[i][band]) for i in range(4)],
        rec_vs_U=dB((Urec - U)[band], U[band]),
        psi_alvos_rad=[[round(float(v), 2) for v in -2 * np.pi * dC0[i] / cfg.system.wl] for i in range(4)],
        h_alvos=[round(float(v), 1) for v in hp])


out["plano"] = cena(plano)
out["montanha"] = cena(mont)

# (b) um alvo só em x = 2600 m, h = 0: U/K é a rampa da posição
cfg, tr = setup(np.array([2600.0]), plano)
f, K, H = filtros(cfg, tr); band = np.abs(f) < cfg.abw / 2
U = np.fft.fft(S.reconstruct(cfg, tr, S.generate_channels(cfg, tr)))
o = np.argsort(f[band]); G = (U[band] / K[band])[o]; fb = f[band][o]
out["um_alvo"] = dict(mod_medio=round(float(np.abs(G).mean()), 3), desvio_rel=round(float(np.abs(G).std() / np.abs(G).mean()), 4),
                      inclinacao_amostras=round(float(-np.polyfit(fb, np.unwrap(np.angle(G)), 1)[0] * cfg.prf / (2 * np.pi)), 2),
                      esperado=round(2600 / (cfg.system.vs / cfg.prf), 2))
json.dump(out, open(os.path.join(HERE, "numeros_K_Ki.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
