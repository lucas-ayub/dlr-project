"""Exemplo mínimo: um alvo na encosta de uma montanha, reconstruído de três maneiras
(padrão, joint nos blocos e joint na linha inteira)."""
from dataclasses import replace
import numpy as np
import sata1d as S
from sata1d.config import ArrayGeometry

# 1) cena e configuração: um alvo na encosta de uma montanha gaussiana de 1000 m
terreno = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
x_alvo = 1000.0
alvos = [(x_alvo, float(terreno(x_alvo)))]                    # (x [m], h [m]): na encosta, h = 801 m
cfg = S.make_config(alvos, bxt_max=0.0, tx_rx_channel0=True)  # canal 0 no transmissor
cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.array([0.0, 134.9, 20.5, 8.3])))
tr = S.build_platform_tracks(cfg)                             # trilhas do Tx e dos Rx

# 2) dados: os 4 canais a PRF_op e a referência monostática a PRF cheia
ch = S.generate_channels(cfg, tr)                             # [4, Na_ch]
ref = S.generate_reference(cfg, tr)                           # [Na]

# 3) P3: o DEM nas duas grades (os métodos só recebem o terreno, não os alvos)
dem_canais = S.channel_maps(cfg, tr, terreno)                 # [4, Na_ch]: grade de cada canal
dem_linha = S.line_maps(cfg, tr, terreno)                     # [4, Na]: grade da linha cheia

# 4) reconstruções: padrão, joint nos blocos e joint na linha inteira
y_pad = S.reconstruct(cfg, tr, ch)
y_blc = S.reconstruct_joint(cfg, tr, ch, maps=dem_canais, sata_osf=4, occupied=True)
y_lin, info = S.reconstruct_joint_line(cfg, tr, ch, dem_linha, return_info=True)

# 5) F3: compressão em azimute com a resposta do ponto de referência
h_ref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
casado = np.conj(np.fft.fft(h_ref))
foco = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * casado), cfg.Na // 2))
pico = foco(ref).max()
t = (np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf
amb = np.abs(np.abs(t - x_alvo / cfg.system.vs) - 0.81) < 0.02    # perto da 1a ambiguidade (+-X/v)
linhas = (("referência", ref), ("padrão", y_pad), ("joint nos blocos", y_blc),
          ("joint na linha inteira", y_lin))
for nome, y in linhas:
    print(f"{nome:22s} ambiguidade a uma dobra: {20 * np.log10(foco(y)[amb].max() / pico):6.1f} dB")
nao_explicado = 20 * np.log10(info["unexplained"])
print(f"iterações: {info['iterations']}; parte dos canais não explicada: {nao_explicado:.1f} dB")
