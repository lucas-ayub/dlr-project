"""
PDF with the block map with copies (footprint) and the look-up table on the same page, for every case of
combined_main_results_tx_rx.pdf: upper IRF panel = SATA with the block map with copies, lower IRF panel = SATA with
the look-up table (in the place of no SATA). Channel 0 on the transmitter.

  cases_compare/    run_sweep_lut_cases.py --map footprint --full-line --compare-maps   (through tx_rx.py)
  terrain_compare/  run_terrain_lut_cases.py --map footprint --full-line --compare-maps (through tx_rx.py)

    python make_compare.py            # writes ../combined_footprint_vs_lut_tx_rx.pdf
"""
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)
c = lambda n: os.path.join(HERE, "cases_compare", n + "_footprint.png")
t = lambda cid, n: os.path.join(HERE, "terrain_compare", f"{cid}_N{n:02d}_footprint.png")
PAGES = [
    # along-track ramps
    c("RA-SS-012_ramp_N2_a40"), c("RA-012_ramp_N2_a40"), c("RA-049_ramp_N7_a20"), c("RA-NN-036_ramp_N5_a20"),
    c("RA-NN-038_ramp_N5_a40"), c("RA-LN-038_ramp_N5_a40"), c("MP-R13_ramp_MP-R13"), c("MP-R25_ramp_MP-R25"), c("MP-R49_ramp_MP-R49"),
    # random geometries
    c("RH-017_random_heights_N13_hmax1000_seed0"), c("CL-012_cliff_N9_dh500"), c("SK-007_spike_N9_dh1000"), c("ME-005_mesa_N9_dh500"),
    c("RH-NN-005_random_heights_N5_hmax1000_seed0"), c("PW-NN-002_piecewise_N5_seed1"), c("ZI-NN-012_zigzag_N5_a40"),
    c("CL-NN-005_cliff_N5_dh500"), c("SK-NN-003_spike_N5_dh1000"), c("ME-NN-002_mesa_N5_dh500"),
    c("RH-LN-005_random_heights_N5_hmax1000_seed0"), c("MP-H13_ramp_MP-H13"), c("MP-H25_ramp_MP-H25"),
    # sine / cosine terrains
    t("SIN-B100-A100-P50", 1), t("SIN-B100-A100-P50", 5), t("SIN-B100-A100-P50", 6), t("SIN-B100-A100-P50", 31),
    t("SIN-B500-A100-P50", 6), t("COS-B80-A80-P20", 5), t("COS-B80-A80-P20", 6), t("COS-B40-A40-P5", 6), t("COS-B40-A40-P5", 7),
    t("SIN-B40-A40-P5", 31), t("REF-A100-P50", 1),
    # complete scenes: one scatterer (random phase) in every cell over +-8 km
    c("CO-R100_ramp_CO-R100"), c("CO-R1000_ramp_CO-R1000"), c("CO-HILL_ramp_CO-HILL"), c("CO-WAVE_ramp_CO-WAVE"),
]


def main():
    import img2pdf
    from PIL import Image
    tmp = tempfile.mkdtemp(); pngs = []
    for n, page in enumerate(PAGES):
        im = Image.open(page); rgb = os.path.join(tmp, f"p{n:03d}.png")
        im.convert("RGB").save(rgb, dpi=im.info.get("dpi", (95, 95)), optimize=True); pngs.append(rgb)
    dst = os.path.join(FIG, "combined_footprint_vs_lut_tx_rx.pdf")
    with open(dst, "wb") as fh:
        fh.write(img2pdf.convert(pngs))
    print("wrote", dst, len(pngs), "pages", round(os.path.getsize(dst) / 2 ** 20, 1), "MiB")


if __name__ == "__main__":
    main()
