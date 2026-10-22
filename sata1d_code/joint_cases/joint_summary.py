"""Summary page of run_joint_cases.py for each output folder (one row per case, highest ambiguity of every method
[dB]) and one PDF with every configuration: summary page, then one page per case.
Run from sata1d_code after run_joint_cases.py:
    python joint_cases/joint_summary.py joint_cases/bxt100 joint_cases/bxt500 joint_cases/bxt1000 --pdf joint_cases/joint_cases.pdf"""
import argparse
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                     # run_joint_cases
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # sata1d_code
import run_joint_cases as R

COLS = [("ref", "reference"), ("none", "no SATA"), ("whole", "SATA whole band\n(block map)"), ("sub", "SATA per sub-band\n(block map)"),
        ("joint_full", "joint\nfull system"), ("joint_occ", "joint\noccupied"), ("oracle", "oracle")]


def summary(out_dir):
    res = json.load(open(os.path.join(out_dir, "results.json")))
    bxt = res.pop("_bxt", None)
    ids = [c for c in R.CASES if c in res]
    cell = [[f"{res[c][k]:.1f}" for k, _ in COLS] + [f"{res[c]['joint_occ'] - res[c]['whole']:+.1f}",
                                                     f"{res[c]['joint_occ'] - res[c]['joint_full']:+.1f}"] for c in ids]
    fig, ax = plt.subplots(figsize=(16, 0.33 * len(ids) + 1.6)); ax.axis("off")
    tb = ax.table(cellText=cell, rowLabels=ids, colLabels=[l for _, l in COLS] + ["joint occupied\n- whole band", "joint occupied\n- joint full"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.35)
    for (r, c), t in tb.get_celld().items():
        if r == 0:
            t.set_height(t.get_height() * 1.9); t.set_text_props(fontweight="bold")
        if r > 0 and c >= len(COLS):
            d = float(cell[r - 1][c])
            t.set_facecolor("#d9efe3" if d <= -1 else ("#f8dcdc" if d >= 1 else "white"))
        if r > 0 and c == 5:
            t.set_text_props(fontweight="bold")
    bt = "" if bxt is None else " | $b_{xt}$ = [" + ", ".join(f"{b:.1f}" for b in bxt) + "] m"
    ax.set_title(f"Highest ambiguity [dB re. the reference peak] of every case{bt}\n"
                 "last two columns: green = joint occupied at least 1 dB lower (better), red = at least 1 dB higher (worse)", fontsize=11)
    fig.text(0.5, 0.01, "Nrx = 4, PRF = 2000 Hz, Ba = 1235 Hz, channel 0 is the transmitter (bat = bxt = 0). Joint maps: nearest "
             "target (point-target scenes), look-up table from the terrain (terrains).", ha="center", fontsize=8.5, color="0.35")
    png = os.path.join(out_dir, "00_summary.png")
    fig.savefig(png, dpi=95, bbox_inches="tight"); plt.close(fig)
    return [png] + [os.path.join(out_dir, f"{c}_joint.png") for c in ids]


def bundle(pngs, out):
    """Multi-page PDF, one page per PNG, JPEG-compressed (keeps the file small)."""
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in pngs]
    ims[0].save(out, save_all=True, append_images=ims[1:], resolution=95, quality=78)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["joint_cases/bxt100"])
    ap.add_argument("--pdf", default=None)
    a = ap.parse_args()
    pngs = sum((summary(d) for d in a.dirs), [])
    if a.pdf:
        print("wrote", bundle(pngs, a.pdf))


if __name__ == "__main__":
    main()
