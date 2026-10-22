"""Exemplo mínimo: um alvo numa montanha, reconstrução padrão e joint com o DEM."""
from dataclasses import replace
import numpy as np
import sata1d as S
from sata1d.config import ArrayGeometry

# 1) cena e configuração: um alvo no pico de uma montanha gaussiana de 1000 m
terreno = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
alvos = [(0.0, float(terreno(0.0)))]                             # (x [m], h [m]): no pico, h = 1000 m
cfg = S.make_config(alvos, bxt_max=0.0, tx_rx_channel0=True)     # canal 0 no transmissor
cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.array([0.0, 134.9, 20.5, 8.3])))
tr = S.build_platform_tracks(cfg)                                # trilhas do Tx e dos Rx

# 2) dados: os 4 canais a PRF_op e a referência monostática a PRF cheia
ch = S.generate_channels(cfg, tr)                                # [4, Na_ch]
ref = S.generate_reference(cfg, tr)                              # [Na]

# 3) P2: a tabela do DEM, uma por canal
maps = np.array([S.terrain_lookup_table(cfg, tr, i, terreno, step=4) for i in range(cfg.Nrx)])

# 4) reconstruções: padrão e joint com dobras ocupadas
y_pad = S.reconstruct(cfg, tr, ch)
y_jnt = S.reconstruct_joint(cfg, tr, ch, maps=maps, sata_osf=4, occupied=True)

# 5) F3: compressão em azimute com a resposta do ponto de referência
h_ref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
foco = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * np.conj(np.fft.fft(h_ref))), cfg.Na // 2))
pico = foco(ref).max()
t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
amb = np.abs(np.abs(t) - 0.81) < 0.02                            # perto da 1a ambiguidade (+-X/v)
for nome, y in (("referência", ref), ("padrão", y_pad), ("joint", y_jnt)):
    print(f"{nome:10s} ambiguidade a uma dobra: {20 * np.log10(foco(y)[amb].max() / pico):6.1f} dB")
