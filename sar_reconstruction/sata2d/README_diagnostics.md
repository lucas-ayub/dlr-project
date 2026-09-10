# Scripts de diagnóstico (set. 2026)

Complementam os `run_sata_*`. Rodar de `sar_reconstruction/`, ex.
`python -m sata2d.run_irf2d`.

| arquivo | o que faz |
|---|---|
| `run_irf2d.py`   | **IRF 2-D** — contorno range × azimute + os dois cortes 1-D, no estilo da Fig. 2.9 de Sakar |
| `run_check.py`   | reconstrói (sem SATA / sub-banda), mede a razão erro/sinal, grava `plots/cache_check.npz` |
| `plot_esr.py`    | 4 painéis **error budget** a partir do cache: mapa 2-D erro/sinal, corte vs Doppler, IRF de azimute, zoom do lóbulo |
| `run_axes_2d.py` | 4 painéis **de eixos**: IRF de azimute (slow time), zoom em metros, IRF de range (fast time), espectro 2-D |
| `run_c1c2.py`    | resíduos `dC0/dC1/dC2` em fase + reconstrução "oráculo" (filtro com a altura verdadeira) |
| `run_bxt_test.py`| varredura de `bxt` com filtro oráculo, isolando o desalinhamento em range |

## `run_irf2d.py` — duas coisas que fazem a figura funcionar

**Foco por filtro casado 2-D.** `img = IFFT2(FFT2(x) · conj(FFT2(ref)))`, com
`ref` o sinal monostático ideal do mesmo alvo. É exato: a RCM (~41 m, 8
células) está contida em `ref` e é compensada sem RCMC aproximado. Focar cada
range bin com o chirp de azimute de um único bin — o atalho óbvio — deixa a
RCM sem corrigir e produz contornos tortos.

**Resolução em range comparável à de azimute.** O preset padrão tem
`res_rg = 30 m` contra `La/2 = 6 m` em azimute: razão 5:1, e a cruz sai
achatada. O default do script é `--res-rg 6`, que iguala as duas e reproduz a
cruz isotrópica da figura de referência; `--swath 300` mantém `Nr = 128`, ou
seja o mesmo custo do preset padrão. `--res-rg 30` mostra o caso original,
com os eixos ajustados sozinhos.

Verificação: para a referência monostática o script devolve
PSLR = −13,3 / −13,5 dB em azimute e range — o valor teórico de janela
retangular (−13,26 dB) —, e IRW(−3 dB) ≈ 4–5 m para 6 m de resolução.

```
python -m sata2d.run_irf2d                  # referência monostática
python -m sata2d.run_irf2d --method sub     # SATA por sub-banda (mono tracejado)
python -m sata2d.run_irf2d --res-rg 30      # preset padrão, anisotrópico
```

## Resultados das medições

- `dC1` e `dC2` são **identicamente nulos** em broadside (razões 2,7e-8 e
  2,4e-9 contra `dC0`). Implementá-los não traria nada.
- Oráculo (fase exata): 96,1 % / −6,27 dB. SATA só com `C0`: 95,4 % /
  −6,39 dB. O modelo de fase já está no teto.
- O resíduo restante é o **desalinhamento em range entre canais** causado por
  `bxt`: 1,47 células no preset. Com `bxt` → 0 o oráculo vai a 99,8 % /
  −19,50 dB.

## Aviso sobre diagnósticos

Não use mapas de `arg(S_rec · conj(S_ref))` neste regime. Com erro/sinal de
+1,18 dB (sem SATA) ou −6,39 dB (com), a diferença de fase é quase uniforme e
o mapa mostra ruído, não estrutura. Use `|S_rec − S_ref| / |S_ref|` em dB.

## Documentação

`docs/coreg.pdf` — dedução completa do termo de co-registro em range
(`docs/coreg.tex` é a fonte).
