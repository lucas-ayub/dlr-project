"""Joint reconstruction on real terrain: azimuth lines cut from SRTM 3'' (about 90 m) over the Sierra Nevada.

Every result is one azimuth line (1-D). A line is a 47 km profile of the DEM; the heights are taken relative to the
mean height of the line (the reference height of the processing). Point targets are placed on the terrain, the four
channels are simulated with the cross-track baselines bxt, and three lines are compared, all focused the same way
and limited to the processed band Ba:

    reference      one monostatic radar at the full PRF (2000 Hz), simulated directly
    no correction  standard multichannel reconstruction (reference-point filters, one height)
    joint          reconstruct_joint_line with the DEM phase of every place (line_maps of the same SRTM profile)

The joint reconstruction only receives the geometry and the DEM, never the targets.

DEM: SRTM 3'' (Farr et al., 2007), file raster/srtm90.tif of github.com/opengeos/data, downloaded on the first run
into data/srtm90.tif next to this script (16 MB, not versioned).

    python run_real_dem.py lines            # the five named lines, five targets each
    python run_real_dem.py lines20          # the same lines with 20 targets every 750 m
    python run_real_dem.py random --n 200   # 200 random lines across the DEM
    python run_real_dem.py figures          # report figures (report/figures/realdem_*.pdf) from the saved results
    python run_real_dem.py all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))           # sata1d_code: the sata1d package
import sata1d as S                                            # noqa: E402
from sata1d.config import ArrayGeometry                       # noqa: E402

DEM_URL = "https://raw.githubusercontent.com/opengeos/data/main/raster/srtm90.tif"
DEM_FILE = os.path.join(HERE, "data", "srtm90.tif")
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "..", "..", "report", "figures")

LINE_LENGTH = 47e3                    # [m]
STEP = 90.0                           # sampling of the profile [m] (the DEM posting)
BXT_A = np.array([0.0, 300.0, 119.8, 100.5])   # baselines of the named lines (largest 300 m)
BXT_B = np.array([0.0, 134.9, 20.5, 8.3])      # second set of the random lines (largest 135 m)
TARGETS5 = np.array([-5000.0, -2400.0, 0.0, 2600.0, 5300.0])   # five targets; no ghost (+-X, +-2X) on another target
TARGETS20 = np.linspace(-7125.0, 7125.0, 20)                   # 20 targets every 750 m
GHOST_HALF = 0.020 * 7688.53706432   # half width of a ghost window: +-20 ms (the metric of the report) [m]

# the five named lines: centre (lat, lon), azimuth of the flight direction (0 = north-south, 90 = west-east)
LINES = {
    "yosemite":   dict(name="Yosemite Valley", lat=37.735, lon=-119.590, az=0.0, dir="north-south"),
    "escarpment": dict(name="Eastern Sierra escarpment", lat=37.200, lon=-118.500, az=90.0, dir="west-east"),
    "highsierra": dict(name="High Sierra, Tuolumne", lat=37.880, lon=-119.360, az=0.0, dir="north-south"),
    "foothills":  dict(name="Western foothills", lat=37.650, lon=-120.250, az=90.0, dir="west-east"),
    "mono":       dict(name="Mono basin", lat=37.900, lon=-119.100, az=0.0, dir="north-south"),
}


# --------------------------------------------------------------------------------------------- the DEM
class DEM:
    """SRTM raster (EPSG:4326) with bilinear sampling at (lat, lon)."""

    def __init__(self, path=DEM_FILE):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading {DEM_URL}")
            urllib.request.urlretrieve(DEM_URL, path)
        import rasterio
        with rasterio.open(path) as r:
            self.h = r.read(1).astype(float)
            self.bounds = r.bounds
            self.res = r.res
        self.lat0, self.lon0 = self.bounds.top, self.bounds.left

    def sample(self, lat, lon):
        from scipy.ndimage import map_coordinates
        row = (self.lat0 - np.asarray(lat)) / self.res[1] - 0.5
        col = (np.asarray(lon) - self.lon0) / self.res[0] - 0.5
        return map_coordinates(self.h, [row, col], order=1, mode="nearest")

    def line(self, lat, lon, az, length=LINE_LENGTH, step=STEP):
        """Profile of a straight line through (lat, lon) with direction az [deg from north]:
        x [m] along the line (0 at the centre), heights [m], and the lat/lon of every sample."""
        x = np.arange(-length / 2, length / 2 + step / 2, step)
        a = np.deg2rad(az)
        lat_s = lat + x * np.cos(a) / 111320.0
        lon_s = lon + x * np.sin(a) / (111320.0 * np.cos(np.deg2rad(lat)))
        return x, self.sample(lat_s, lon_s), lat_s, lon_s

    def inside(self, lat, lon, az, margin=0.01):
        _, _, la, lo = self.line(lat, lon, az, step=LINE_LENGTH / 2)
        b = self.bounds
        return (la.min() > b.bottom + margin and la.max() < b.top - margin
                and lo.min() > b.left + margin and lo.max() < b.right - margin)


# --------------------------------------------------------------------------------------------- one line
def run_line(x, h, targets, bxt, keep=False):
    """Simulate the channels for targets on the profile (x, h) and reconstruct with and without the DEM.
    Returns the levels [dB] of the strongest ghost of the three methods (and the focused lines if keep)."""
    hrel = h - h.mean()                                        # heights relative to the reference height
    terrain = lambda xx: np.interp(np.asarray(xx, float), x, hrel)
    tg = [(float(t), float(terrain(t))) for t in targets]
    cfg = S.make_config(tg, bxt_max=0.0, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float)))
    tr = S.build_platform_tracks(cfg)
    ch = S.generate_channels(cfg, tr)
    ref = S.generate_reference(cfg, tr)
    y_std = S.reconstruct(cfg, tr, ch)
    dC0 = S.line_maps(cfg, tr, terrain)
    y_jnt, info = S.reconstruct_joint_line(cfg, tr, ch, dC0, return_info=True)

    # focusing: matched filter of the reference point, the same for every method, in the processed band Ba
    f = np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)
    band = np.abs(f) <= cfg.abw / 2
    H = np.conj(np.fft.fft(S.generate_reference(cfg, tr, cfg.scene.ptg[None, :]))) * band
    focus = lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * H), cfg.Na // 2)
    foc = dict(ref=focus(ref), std=focus(y_std), joint=focus(y_jnt))
    pos = S.joint_line.line_positions(cfg)
    peak = np.abs(foc["ref"]).max()
    X = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2) * cfg.system.vs
    win = np.zeros(cfg.Na, bool)
    for t in targets:
        for k in (-2, -1, 1, 2):
            win |= np.abs(pos - (t + k * X)) <= GHOST_HALF
    lev = {k: float(20 * np.log10(np.abs(v[win]).max() / peak)) for k, v in foc.items()}
    out = dict(levels=lev, iterations=info["iterations"], unexplained_db=float(20 * np.log10(info["unexplained"])),
               relief=float(h.max() - h.min()), hmin=float(h.min()), hmax=float(h.max()), X=float(X),
               bxt=[float(b) for b in bxt], targets=[[a, b] for a, b in tg], hmean=float(h.mean()))
    if keep:
        out.update(pos=pos, foc=foc, peak=peak, terrain_x=x, terrain_h=h)
    return out


# --------------------------------------------------------------------------------------------- the runs
def run_named(dem, targets, tag, save_lines=True):
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for key, L in LINES.items():
        t0 = time.time()
        x, h, la, lo = dem.line(L["lat"], L["lon"], L["az"])
        r = run_line(x, h, targets, BXT_A, keep=True)
        lev = r["levels"]
        print(f"{tag:8s} {L['name']:28s} {r['hmin']:5.0f}-{r['hmax']:5.0f} m  ref {lev['ref']:6.1f}  "
              f"no correction {lev['std']:6.1f}  joint {lev['joint']:6.1f} dB  ({time.time() - t0:.0f} s)")
        if save_lines:
            dec = slice(None, None, 2)
            np.savez_compressed(os.path.join(OUT, f"{tag}_{key}.npz"), pos=r["pos"][dec],
                                ref=r["foc"]["ref"][dec], std=r["foc"]["std"][dec], joint=r["foc"]["joint"][dec],
                                peak=r["peak"], x=x, h=h, lat=la, lon=lo)
        res[key] = {k: v for k, v in r.items() if k not in ("pos", "foc", "peak", "terrain_x", "terrain_h")}
        res[key].update(L)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def run_random(dem, n=200, seed=1):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    b = dem.bounds
    rows = []
    while len(rows) < n:
        lat, lon, az = rng.uniform(b.bottom, b.top), rng.uniform(b.left, b.right), rng.uniform(0.0, 180.0)
        if not dem.inside(lat, lon, az):
            continue
        x, h, _, _ = dem.line(lat, lon, az)
        bxt = BXT_A if len(rows) % 2 == 0 else BXT_B
        t0 = time.time()
        r = run_line(x, h, TARGETS5, bxt)
        lev = r["levels"]
        rows.append([lat, lon, az, r["hmin"], r["hmax"], bxt.max(), lev["ref"], lev["std"], lev["joint"],
                     r["iterations"], r["unexplained_db"]])
        print(f"{len(rows):4d} relief {r['relief']:5.0f} m  bxt {bxt.max():3.0f}  ref {lev['ref']:6.1f}  "
              f"no correction {lev['std']:6.1f}  joint {lev['joint']:6.1f} dB  ({time.time() - t0:.0f} s)", flush=True)
        np.savetxt(os.path.join(OUT, "random_lines.csv"), np.array(rows), delimiter=",", fmt="%.4f",
                   header="lat,lon,az_deg,hmin_m,hmax_m,bxt_max_m,ref_db,no_correction_db,joint_db,iterations,unexplained_db")
    return np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lines", "lines20", "random", "figures", "all"])
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dem = DEM() if a.what != "figures" else None
    if a.what in ("lines", "all"):
        run_named(dem, TARGETS5, "lines5")
    if a.what in ("lines20", "all"):
        run_named(dem, TARGETS20, "lines20")
    if a.what in ("random", "all"):
        run_random(dem, a.n)
    if a.what in ("figures", "all"):
        import real_dem_figures
        real_dem_figures.main()
