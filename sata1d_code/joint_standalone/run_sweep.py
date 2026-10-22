"""
Sweep of the joint reconstruction on the whole line (standalone: only joint1d/ and sweep_page.py, both next to this file).

Choose the families of scenes in FAMILIES (below). For every scene and every set of cross-track baselines it simulates
the 4 channels, runs the standard reconstruction (no SATA) and the joint (DEM only), and writes in plots/sweep_<names>/:
    pages/<case>.png     one page per case (IRF, scene in 2D and 3D, channels)
    numbers/<case>.json  the numbers of the case
    table.txt            highest ambiguity of every case
    sweep_<names>.pdf    all pages, after a summary page

    python run_sweep.py                  # runs what is missing and then writes the table and the PDF
    python run_sweep.py --budget 160     # stops starting new cases after 160 s (run again to continue)
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint1d as J
from joint1d.config import ArrayGeometry
import sweep_page as P

VERSION = 3   # cases saved with another version are computed again

# ------------------------------------------------------------------------------------------------ which scenes
# FAMILIES = ["ramps", "sine_cosine", "realistic", "extreme"]          # all the options
FAMILIES = ["extreme"]

def random_baselines(nrx, bmax, seed=0):
    """Channel 0 on the transmitter; the others random in (0, bmax], the largest exactly bmax (as the project sweeps)."""
    u = np.random.default_rng(seed).uniform(0.1, 1.0, nrx)
    u[0] = 0.0
    return [float(v) for v in (bmax * u / u.max()).round(1)]


BASELINES = {  # tag -> (cross-track baselines [m] of channels 0..Nrx-1, description); channel 0 is the transmitter
    "u100": ([0.0, 27.0, 4.1, 1.7], "random, U(0, 100 m)"),
    "u500": ([0.0, 134.9, 20.5, 8.3], "random, U(0, 500 m)"),
    "s50": ([0.0, 50.0, 100.0, 150.0], "steps of 50 m"),
    "pm50": ([0.0, -50.0, 50.0, 100.0], "negative and positive"),
    "N2b300": (random_baselines(2, 300.0), "random, largest 300 m, $N_{rx}$ = 2"),
    "N3b300": (random_baselines(3, 300.0), "random, largest 300 m, $N_{rx}$ = 3"),
    "N6b300": (random_baselines(6, 300.0), "random, largest 300 m, $N_{rx}$ = 6"),
    "N8b300": (random_baselines(8, 300.0), "random, largest 300 m, $N_{rx}$ = 8"),
    # extreme cross-track baselines (used by the family "extreme", see FAMILY_BASELINES)
    "N4b300": (random_baselines(4, 300.0), "random, largest 300 m, $N_{rx}$ = 4"),
    "x300": ([0.0, 300.0, 150.0, -250.0], "large, mixed signs (300, 150, -250 m)"),
    "s100": ([0.0, 100.0, 200.0, 300.0], "steps of 100 m (up to 300 m)"),
}
DEFAULT_BASELINES = ["u100", "u500", "s50", "pm50", "N2b300", "N3b300", "N6b300", "N8b300"]
FAMILY_BASELINES = {  # family -> baseline tags (families not listed use DEFAULT_BASELINES)
    "extreme": ["u500", "N2b300", "N4b300", "N8b300", "x300", "s100"],
}


# ------------------------------------------------------------------------------------------------ terrains
def random_terrain(seed, relief, corr, half=80e3, dx=20.0, region=22e3):
    """Smooth random terrain: white noise filtered by a Gaussian (correlation length corr), scaled so that
    min -> max over |x| <= region is 0 -> relief (the same terrains as the sweeps of the project)."""
    g = np.random.default_rng(seed)
    x = np.arange(-half, half + dx, dx)
    k = np.fft.rfftfreq(x.size, dx)
    h = np.fft.irfft(np.fft.rfft(g.standard_normal(x.size)) * np.exp(-2.0 * (np.pi * 0.5 * corr * k) ** 2), x.size)
    m = np.abs(x) <= region
    h = (h - h[m].min()) * relief / np.ptp(h[m])
    return lambda xq: np.interp(np.asarray(xq, float), x, h)


def ramp(slope_deg, length):
    """Flat at h = 0, ramp of slope_deg over [-length/2, length/2], flat on top after."""
    t = np.tan(np.radians(slope_deg))
    return lambda x: t * np.clip(np.asarray(x, float) + length / 2.0, 0.0, length)


def sinusoid(mean, amp, period, kind):
    f = np.sin if kind == "sin" else np.cos
    return lambda x: mean + amp * f(2 * np.pi * np.asarray(x, float) / period)


def spaced(n, d):
    return (np.arange(n) - (n - 1) / 2.0) * d


def steepest(terrain, half=3000.0):
    """One target on the steepest point of the terrain within +-half (as the one-target cases of the project sweep)."""
    x = np.arange(-half, half + 1.0, 10.0)
    return np.array([x[int(np.argmax(np.abs(np.gradient(terrain(x), x))))]])


def dist(d):
    return f"{d:g} m" if d < 1000 else f"{d / 1e3:g} km"


TERRAINS = {  # family -> list of (id, h(x), description)
    "ramps": [
        ("ramp3", ramp(3.0, 12000.0), "along-track ramp of 3$^\\circ$, 12 km (0 -> 629 m)"),
        ("ramp8", ramp(8.0, 12000.0), "along-track ramp of 8$^\\circ$, 12 km (0 -> 1686 m)"),
        ("ramp15", ramp(15.0, 3000.0), "along-track ramp of 15$^\\circ$, 3 km (0 -> 804 m)"),
    ],
    "sine_cosine": [
        ("sin50km", sinusoid(100.0, 100.0, 50e3, "sin"), "h = 100 + 100 sin(2$\\pi$ x / 50 km)"),
        ("cos20km", sinusoid(80.0, 80.0, 20e3, "cos"), "h = 80 + 80 cos(2$\\pi$ x / 20 km)"),
        ("cos5km", sinusoid(40.0, 40.0, 5e3, "cos"), "h = 40 + 40 cos(2$\\pi$ x / 5 km)"),
        ("sin5km", sinusoid(40.0, 40.0, 5e3, "sin"), "h = 40 + 40 sin(2$\\pi$ x / 5 km)"),
    ],
    "realistic": [
        ("hills", random_terrain(3, 300.0, 1500.0), "random hills, relief 300 m, correlation length 1.5 km"),
        ("mountains", random_terrain(3, 1000.0, 3000.0), "random mountains, relief 1000 m, correlation length 3 km"),
        ("escarpment", lambda x: 250.0 * (1.0 + np.tanh(np.asarray(x, float) / 500.0)), "escarpment 0 -> 500 m, 1 km wide"),
        ("mountain", lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2), "isolated mountain 1000 m (Gaussian, $\\sigma$ = 1.5 km)"),
        ("plateau_n", lambda x: 250.0 * (np.tanh((np.asarray(x, float) + 1200.0) / 500.0) - np.tanh((np.asarray(x, float) - 1200.0) / 500.0)),
         "plateau 500 m high, 2.4 km wide"),
        ("plateau_w", lambda x: 250.0 * (np.tanh((np.asarray(x, float) + 3900.0) / 500.0) - np.tanh((np.asarray(x, float) - 3900.0) / 500.0)),
         "plateau 500 m high, 7.8 km wide"),
    ],
    "extreme": [  # relief of the order of 3 km, and two 1 km scenes to see the large baselines alone
        ("mountain3k", lambda x: 3000.0 * np.exp(-0.5 * (np.asarray(x, float) / 3000.0) ** 2), "isolated mountain 3000 m (Gaussian, $\\sigma$ = 3 km)"),
        ("mountains3k", random_terrain(3, 3000.0, 4000.0), "random mountains, relief 3000 m, correlation length 4 km"),
        ("escarpment3k", lambda x: 1500.0 * (1.0 + np.tanh(np.asarray(x, float) / 1500.0)), "escarpment 0 -> 3000 m, 3 km wide"),
        ("plateau3k", lambda x: 1500.0 * (np.tanh((np.asarray(x, float) + 3000.0) / 1000.0) - np.tanh((np.asarray(x, float) - 3000.0) / 1000.0)),
         "plateau 3000 m high, 6 km wide"),
        ("mountain", lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2), "isolated mountain 1000 m (Gaussian, $\\sigma$ = 1.5 km)"),
        ("escarpment", lambda x: 250.0 * (1.0 + np.tanh(np.asarray(x, float) / 500.0)), "escarpment 0 -> 500 m, 1 km wide"),
    ],
}
TARGETS = [  # (id, number of targets, spacing [m]): every terrain of the chosen families with these layouts
    ("1", 1, None),          # one target, on the steepest point of the terrain within +-3 km
    ("5", 5, 2600.0),
    ("25", 25, 500.0),
    ("97", 97, 125.0),       # many targets (random phases), 12 km of scene
]
# more layouts, the same as the realistic cases of the project sweep (sweep_joint_all_1d_realistic, parts 2 and 3):
# family -> list of (terrain id, number of targets, spacing [m], random phases of the targets)
EXTRA_LAYOUTS = {
    "realistic": [
        ("escarpment", 9, 200.0, False),     # few_escarpment_9
        ("plateau_n", 9, 500.0, False),      # few_plateau_9
        ("mountains", 13, 200.0, False),     # many_mountains_13
        ("mountains", 25, 500.0, False),     # many_mountains_25
        ("mountains", 49, 250.0, True),      # many_mountains_49
        ("hills", 49, 250.0, True),          # many_hills_49
        ("escarpment", 25, 250.0, True),     # many_escarpment_25
        ("mountain", 33, 250.0, True),       # many_peak_33
    ],
}


EXCLUDE = ["hills_1"]   # scenes left out of the sweep (case id = <scene>_<baseline tag>)


def cases():
    out = []
    for fam in FAMILIES:
        terr = {tid: (terrain, desc) for tid, terrain, desc in TERRAINS[fam]}
        for tid, (terrain, desc) in terr.items():
            for nid, n, d in TARGETS:
                xs = steepest(terrain) if n == 1 else spaced(n, d)
                for btag in FAMILY_BASELINES.get(fam, DEFAULT_BASELINES):
                    out.append(dict(id=f"{tid}_{nid}_{btag}", family=fam, terrain=terrain, desc=desc,
                                    xs=xs, d=d, btag=btag, random_phase=n > 13))
        for tid, n, d, rp in EXTRA_LAYOUTS.get(fam, []):
            terrain, desc = terr[tid]
            for btag in FAMILY_BASELINES.get(fam, DEFAULT_BASELINES):
                out.append(dict(id=f"{tid}_{n}_{d:g}m{'_rp' if rp else ''}_{btag}", family=fam, terrain=terrain, desc=desc,
                                xs=spaced(n, d), d=d, btag=btag, random_phase=rp))
    return [c for c in out if not any(c["id"] == f"{e}_{c['btag']}" for e in EXCLUDE)]


# ------------------------------------------------------------------------------------------------ one case
def run_case(c, out_dir):
    xs, terrain = c["xs"], c["terrain"]
    n = len(xs)
    amps = np.exp(2j * np.pi * np.random.default_rng(1).uniform(size=n)) if c["random_phase"] else None   # random phases of the targets
    bxt, kind = BASELINES[c["btag"]]
    cfg = J.make_config(list(zip(xs, terrain(xs))), nrx=len(bxt), bxt_max=0.0, tx_rx_channel0=True, amplitudes=amps)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float)))
    tr = J.build_platform_tracks(cfg)
    t0 = time.time()
    ch = J.generate_channels(cfg, tr)
    reference = J.generate_reference(cfg, tr)
    no_sata = J.reconstruct(cfg, tr, ch)
    dC0 = J.line_maps(cfg, tr, terrain)                         # DEM only
    joint, info = J.reconstruct_joint_line(cfg, tr, ch, dC0, return_info=True)
    Hf = np.conj(np.fft.fft(J.generate_reference(cfg, tr, cfg.scene.ptg[None, :])))
    focus = lambda s: np.abs(np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2))
    env = {"Reference": focus(reference), "no SATA": focus(no_sata), "joint final version": focus(joint)}
    pk = env["Reference"].max()
    amb = {k: P.ambiguity_level(cfg, e, pk, xs) for k, e in env.items()}
    info = dict(iterations=int(info["iterations"]), unexplained_dB=float(20 * np.log10(info["unexplained"])))
    label = (f"{c['desc']}, 1 target at x = {xs[0] / 1e3:g} km (steepest point)" if n == 1 else
             f"{c['desc']}, {n} targets {dist(c['d'])} apart") + (" (random phases)" if c["random_phase"] else "")
    P.page(os.path.join(out_dir, "pages", c["id"] + ".png"), cfg, terrain, xs, label, kind, env, amb, info)
    res = dict(case=c["id"], family=c["family"], label=label, baseline=c["btag"], bxt=list(map(float, bxt)), n_targets=n,
               nrx=len(bxt),
               amb_dB=amb, joint=info, seconds=round(time.time() - t0, 1), version=VERSION)
    json.dump(res, open(os.path.join(out_dir, "numbers", c["id"] + ".json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------------------------------------ summary
def summary(out_dir, name, results):
    keys = ["Reference", "no SATA", "joint final version"]

    def stats(rs):
        gap = np.array([r["amb_dB"]["joint final version"] - r["amb_dB"]["Reference"] for r in rs])
        return (f"{len(rs)} cases | median: " + " | ".join(f"{k} {np.median([r['amb_dB'][k] for r in rs]):.1f}" for k in keys) +
                f" | joint within 3 dB of the reference in {int(np.sum(gap <= 3))}/{len(gap)}, within 10 dB in {int(np.sum(gap <= 10))}/{len(gap)}")

    lines = [f"{'case':30s} {'Nrx':>3s} {'Reference':>10s} {'no SATA':>9s} {'joint':>8s}  joint-ref  (highest ambiguity, dB)"]
    for r in results:
        a = r["amb_dB"]
        lines.append(f"{r['case']:30s} {r['nrx']:3d} {a['Reference']:10.1f} {a['no SATA']:9.1f} {a['joint final version']:8.1f}  "
                     f"{a['joint final version'] - a['Reference']:+8.1f}")
    lines.append("")
    lines.append("all: " + stats(results))
    for nrx in sorted({r["nrx"] for r in results}):
        lines.append(f"Nrx = {nrx}: " + stats([r for r in results if r["nrx"] == nrx]))
    open(os.path.join(out_dir, "table.txt"), "w").write("\n".join(lines) + "\n")
    sty = {"Reference": dict(color="tab:green", marker="s"), "no SATA": dict(color="tab:red", marker="x"),
           "joint final version": dict(color="tab:purple", marker="o")}
    for nrx in sorted({r["nrx"] for r in results}):
        rs = [r for r in results if r["nrx"] == nrx]
        fig, ax = plt.subplots(figsize=(22, 10.6))
        x = np.arange(len(rs))
        for k in keys:
            ax.plot(x, [r["amb_dB"][k] for r in rs], ls="none", ms=6, label=k, **sty[k])
        ax.set_xticks(x); ax.set_xticklabels([r["case"] for r in rs], rotation=90, fontsize=6.5)
        ax.set_ylabel("highest ambiguity [dB re. reference peak]"); ax.grid(alpha=0.3); ax.legend(fontsize=10)
        ax.set_title(f"Sweep of the joint reconstruction ({', '.join(FAMILIES)}), $N_{{rx}}$ = {nrx}: {stats(rs)}", fontsize=11)
        fig.tight_layout()
        pdf_path = os.path.join(out_dir, f"sweep_{name}_Nrx{nrx}.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig); plt.close(fig)
            for r in rs:
                img = plt.imread(os.path.join(out_dir, "pages", r["case"] + ".png"))
                f2 = plt.figure(figsize=(img.shape[1] / 90, img.shape[0] / 90))
                f2.figimage(img, resize=False); pdf.savefig(f2, dpi=90); plt.close(f2)
        print("saved", pdf_path)
    print("\n".join(lines[-(len({r['nrx'] for r in results}) + 1):]))


def _done(out_dir, c):
    p = os.path.join(out_dir, "numbers", c["id"] + ".json")
    return os.path.exists(p) and json.load(open(p)).get("version") == VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None, help="stop starting new cases after this many seconds")
    a = ap.parse_args()
    name = "_".join(FAMILIES)
    out_dir = os.path.join(HERE, "plots", f"sweep_{name}")
    os.makedirs(os.path.join(out_dir, "pages"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "numbers"), exist_ok=True)
    allc = cases()
    start = time.time()
    for k, c in enumerate(allc):
        if _done(out_dir, c):
            continue
        if a.budget is not None and time.time() - start > a.budget:
            done = sum(_done(out_dir, q) for q in allc)
            print(f"stopped by the budget: {done}/{len(allc)} cases done, run again to continue")
            return
        r = run_case(c, out_dir)
        print(f"[{k + 1}/{len(allc)}] {c['id']}: joint {r['amb_dB']['joint final version']:.1f} dB, "
              f"reference {r['amb_dB']['Reference']:.1f} dB ({r['seconds']} s)", flush=True)
    results = [json.load(open(os.path.join(out_dir, "numbers", c["id"] + ".json"))) for c in allc]
    summary(out_dir, name, results)


if __name__ == "__main__":
    main()
