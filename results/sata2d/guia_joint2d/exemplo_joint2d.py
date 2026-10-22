"""Exemplo mínimo do joint em 2-D: um alvo na encosta de uma montanha, quatro canais (rodar a partir desta pasta)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sar_reconstruction"))
from sata2d.arrays import make_params_dpca
from sata2d.geometry import build_tracks_3d, generate_reference_3d, generate_channels_3d, CoeffTable3D
from sata2d.reconstruction import range_compress, reconstruct_subband_2d
from sata2d.joint2d import dem_maps, reconstruct_joint_2d, Model2D, processed_band
from sata2d.run_sweep_joint2d import envelopes, ambiguity_level

# 1) o terreno (o DEM): altura [m] em função da posição em azimute x e da distância no solo dy ao ponto de referência
terreno = lambda x, dy: 1000.0 * np.exp(-0.5 * ((x / 1500.0) ** 2 + (dy / 4000.0) ** 2))

# 2) o sistema: quatro canais na condição DPCA, linhas de base cross-track, um alvo na encosta da montanha
p = make_params_dpca(Nrx=4, specs=(), r0_frac=0.6)
p.bxt = np.array([0.0, 300.0, 119.8, 100.5])
x_alvo, dy_alvo = 1500.0, 1000.0
p.extra_offsets = ((x_alvo, dy_alvo, float(terreno(x_alvo, dy_alvo))),)
tr = build_tracks_3d(p)

# 3) os dados: canais a PRF_op e referência monostática na PRF cheia, comprimidos em range
ch = range_compress(generate_channels_3d(p, tr), p.cd, p.rbw, p.rsf, axis=2)       # [Nrx, Na_ch, Nr]
ref = range_compress(generate_reference_3d(p, tr), p.cd, p.rbw, p.rsf, axis=1)     # [Na, Nr]

# 4) o DEM em geometria de radar e as duas reconstruções
dC0, h = dem_maps(p, tr, terreno)                                                  # [Nrx, Na, 2 Nr] e [Na, 2 Nr]
M = Model2D(p, tr, dC0)
joint, info = reconstruct_joint_2d(p, tr, ch, dC0, model=M, return_info=True)      # [Na, Nr]
padrao = reconstruct_subband_2d(p, tr, ch, CoeffTable3D(p, tr, n_nodes=16), use_sata=False, verbose=False)

# 5) a medida: IRF de azimute, as três curvas na mesma faixa
lo, hi = info["band"]
env = envelopes(p, tr, dict(ref=processed_band(p, M.to_common_band(ref, lo, hi)), padrao=M.to_common_band(padrao, lo, hi), joint=joint))
print(f"alvo em x = {x_alvo:.0f} m, dy = {dy_alvo:.0f} m, h = {p.extra_offsets[0][2]:.0f} m; dC0 do canal 1 até {dC0[1].min():.2f} m")
for nome, k in (("referência", "ref"), ("padrão (no SATA)", "padrao"), ("joint em 2-D", "joint")):
    print(f"{nome:18s} maior ambiguidade: {ambiguity_level(p, env[k], env['ref'].max(), p.points[:, 0]):6.1f} dB")
print(f"iterações: {info['iterations']}; parte dos canais não explicada: {20 * np.log10(info['unexplained']):.1f} dB; "
      f"faixa de range mantida: {100 * np.median(np.clip(hi - lo, 0, None)) / p.rbw:.0f} % (pixel mediano)")
