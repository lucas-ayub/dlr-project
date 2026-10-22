# Results

The generated outputs (PDF reports, figures, CSV/JSON numbers, caches) are not versioned. This folder only keeps the
scripts that assemble the report PDFs from those outputs:

```
sata1d/figures/combined_main_results/make_combined.py          combined PDF of the main SATA 1D results
sata1d/figures/combined_main_results_tx_rx/                    same, with channel 0 on the transmitter
    make_combined.py, make_compare.py, sweep_v4_tx_rx.py, tx_rx.py
sata2d/guia_joint2d/                                           data, figures and minimal example of the 2-D joint guide
    dados_guia.py, graficos_guia.py, exemplo_joint2d.py
```

Each script writes its outputs next to itself (PYTHONPATH must contain `sata1d_code` for the SATA 1D ones).
