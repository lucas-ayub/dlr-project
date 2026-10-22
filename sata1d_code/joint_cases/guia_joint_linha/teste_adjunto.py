"""Três testes rápidos do modelo do joint na linha inteira
(rodar com PYTHONPATH apontando para o pacote sata1d)."""
import numpy as np
import sata1d as S
from sata1d.joint_line import LineModel

terreno = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
d = S.make_config([(0.0, 0.0)]).system.vs / 2000.0           # passo da linha cheia: 3,844 m
x_alvo = 260 * d                                             # um alvo exatamente sobre a amostra 260
cfg = S.make_config([(x_alvo, float(terreno(x_alvo)))], bxt_max=300.0, tx_rx_channel0=True)
tr = S.build_platform_tracks(cfg)
M = LineModel(cfg, tr, S.line_maps(cfg, tr, terreno))
rng = np.random.default_rng(1)
cplx = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

# 1) o adjunto é o adjunto: <A a, b> = <a, A^H b>
a, b = cplx(cfg.Na), cplx(cfg.Nrx, cfg.Na_ch)
lhs, rhs = np.vdot(M.forward(a), b), np.vdot(a, M.adjoint(b))
print(f"teste do adjunto: {abs(lhs - rhs) / abs(lhs):.1e}")

# 2) o modelo reproduz os canais simulados: A g = canais, com g = um ponto na posição do alvo
ch = S.generate_channels(cfg, tr)
g = np.zeros(cfg.Na, complex)
g[260] = 1.0                                                 # ordem de atraso: índice = x / d
erro = np.linalg.norm(M.forward(g) - ch) / np.linalg.norm(ch)
print(f"modelo contra o simulador: {20 * np.log10(erro):.1f} dB")

# 3) em terreno plano e com DEM nulo, o joint na linha inteira é a reconstrução padrão
cfg0 = S.make_config([(x_alvo, 0.0)], bxt_max=300.0, tx_rx_channel0=True)
tr0 = S.build_platform_tracks(cfg0)
ch0 = S.generate_channels(cfg0, tr0)
y_linha = S.reconstruct_joint_line(cfg0, tr0, ch0, np.zeros((cfg0.Nrx, cfg0.Na)))
y_padrao = S.reconstruct(cfg0, tr0, ch0)
dif = np.linalg.norm(y_linha - y_padrao) / np.linalg.norm(y_padrao)
print(f"DEM nulo contra a reconstrução padrão: {dif:.1e}")
