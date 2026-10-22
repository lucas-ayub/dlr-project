"""
The cases of sweep_sata.py (ramps, sines / cosines and REF-A100-P50) with the final joint (joint reconstruction over the
whole line, the DEM applied at every position: sata1d.joint_line.reconstruct_joint_line), on the pages of sweep_final.py:
    top IRF:    Reference + joint
    bottom IRF: Reference + no SATA + SATA whole band + SATA per sub-band
plus a first page with the highest ambiguity of every case. Numbers from sweep_final.compute (audit_dem.compute);
reference, no SATA and the two SATA are checked against sweep_sata/numbers.

    python joint_cases/sweep_sata_joint.py --workers 4      # from sata1d_code (resumable)
    python joint_cases/sweep_sata_joint.py --pdf
    python joint_cases/sweep_sata_joint.py --table
Output: joint_cases/sweep_sata_joint/numbers/<job>.json, pages/<job>.png, sweep_sata_joint.pdf
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sweep_final as SF
import sweep_sata as SW

OUT = os.path.join(HERE, "sweep_sata_joint")
VERSION = 1
METHODS = [("ref", "Reference", "tab:green", "o", 70), ("none", "no SATA", "tab:red", "x", 60),
           ("whole", "SATA whole band", "tab:blue", "s", 46), ("sub", "SATA per sub-band", "tab:orange", "D", 40),
           ("joint_line", "joint", "tab:purple", "^", 70)]


def _paths(sc, tag):
    k = f"{sc}__{tag}"
    return os.path.join(OUT, "numbers", k + ".json"), os.path.join(OUT, "pages", k + ".png")


def _done(sc, tag):
    pj, pp = _paths(sc, tag)
    return os.path.exists(pj) and json.load(open(pj)).get("version_sweep") == VERSION and os.path.exists(pp)


def _job(j):
    sc, tag = j
    t0 = time.time()
    res, d = SF.compute(sc, tag)
    res["version_sweep"] = VERSION
    pj, pp = _paths(sc, tag)
    SF.page(res, d, pp, "joint_sata")
    json.dump(res, open(pj, "w"), indent=1)
    return f"{sc:18s} {tag:7s} ref {res['ref']:6.1f} none {res['none']:6.1f} whole {res['whole']:6.1f} sub {res['sub']:6.1f} " \
           f"joint {res['joint_line']:6.1f} ({time.time() - t0:.0f} s)"


def load():
    out = {}
    for j in SW.jobs():
        pj = _paths(*j)[0]
        if os.path.exists(pj):
            out[j] = json.load(open(pj))
    return out


def same_as_sweep_sata(res):
    """Reference, no SATA and the two SATA must be the numbers of sweep_sata (same code, same cases)."""
    worst = 0.0
    for j, r in res.items():
        q = os.path.join(SW.OUT, "numbers", f"{j[0]}__{j[1]}.json")
        if os.path.exists(q):
            s = json.load(open(q))
            worst = max(worst, max(abs(r[k] - s[k]) for k in ("ref", "none", "whole", "sub")))
    return worst


def summary(res, out_png):
    rows = [(fam, sc, tag) for fam, scenes in SW.FAMILIES for sc, tag in SW.jobs() if sc in scenes and (sc, tag) in res]
    n = len(rows); H = 0.42 * n + 2.2
    fig, ax = plt.subplots(figsize=(15, H))
    fig.subplots_adjust(left=0.38, right=0.97, top=1 - 0.9 / H, bottom=1.1 / H)
    y = np.arange(n)[::-1]
    for k, name, col, m, s in METHODS:
        ax.scatter([res[(sc, tag)][k] for _, sc, tag in rows], y, marker=m, s=s, color=col, label=name, zorder=3,
                   lw=1.6 if m == "x" else 0.6, edgecolor=None if m == "x" else "k")
    for i, (_, sc, tag) in enumerate(rows):
        r = res[(sc, tag)]
        ax.plot([min(r[k] for k, *_ in METHODS), max(r[k] for k, *_ in METHODS)], [y[i]] * 2, color="0.8", lw=0.8, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([SW.short(res[(sc, tag)]) for _, sc, tag in rows], fontsize=8.6)
    split = sum(1 for f, *_ in rows if f == SW.FAMILIES[0][0])
    ax.axhline(n - split - 0.5, color="0.4", lw=1)
    ax.text(1.0, n - 0.6, SW.FAMILIES[0][0], transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=10, weight="bold")
    ax.text(1.0, n - split - 0.7, SW.FAMILIES[1][0], transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=10, weight="bold")
    ax.set_xlabel("highest ambiguity [dB relative to the reference peak]")
    ax.set_ylim(-0.7, n - 0.3); ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", alpha=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=10)
    fig.savefig(out_png, dpi=110); plt.close(fig)


def table():
    res = load()
    for fam, scenes in SW.FAMILIES:
        print(fam)
        for j in SW.jobs():
            r = res.get(j)
            if r and j[0] in scenes:
                print(f"  {SW.short(r):90s} ref {r['ref']:6.1f} none {r['none']:6.1f} whole {r['whole']:6.1f} sub {r['sub']:6.1f} joint {r['joint_line']:6.1f}")
    v = lambda k: np.median([r[k] for r in res.values()])
    print(f"median of {len(res)}: ref {v('ref'):.1f} none {v('none'):.1f} whole {v('whole'):.1f} sub {v('sub'):.1f} joint {v('joint_line'):.1f}")
    print(f"joint better than both SATA: {sum(r['joint_line'] < min(r['whole'], r['sub']) for r in res.values())} / {len(res)}; "
          f"joint within 10 dB of the reference: {sum(r['joint_line'] - r['ref'] <= 10 for r in res.values())} / {len(res)}")
    print(f"largest difference to sweep_sata (ref, none, whole, sub): {same_as_sweep_sata(res):.2e} dB")


def pdf():
    from joint_sweep_summary import bundle
    res = load()
    first = os.path.join(OUT, "pages", "_summary.png")
    summary(res, first)
    pngs, outline = [first], [("Highest ambiguity of every case", 0, 0)]
    for fam, scenes in SW.FAMILIES:
        head = True
        for j in SW.jobs():
            pp = _paths(*j)[1]
            if j[0] not in scenes or j not in res or not os.path.exists(pp):
                continue
            if head:
                outline.append((fam, len(pngs), 0)); head = False
            outline.append((SW.short(res[j]), len(pngs), 1))
            pngs.append(pp)
    return f"{bundle(pngs, os.path.join(OUT, 'sweep_sata_joint.pdf'), outline)} ({len(pngs)} pages)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    if a.table:
        return table()
    if a.pdf:
        return print("wrote", pdf())
    for sub in ("numbers", "pages"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    todo = [j for j in SW.jobs() if not _done(*j)]
    print(len(todo), "jobs to do", flush=True)
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for line in pool.imap_unordered(_job, todo):
                print(line, flush=True)
    else:
        for j in todo:
            print(_job(j), flush=True)
    print("left:", sum(not _done(*j) for j in SW.jobs()), flush=True)


if __name__ == "__main__":
    main()
