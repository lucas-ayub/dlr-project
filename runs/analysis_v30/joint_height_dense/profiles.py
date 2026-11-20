"""Straight azimuth profiles cut from real DEMs (SRTM 90 m of the Sierra Nevada, a 33 m DEM of Oregon)."""
import numpy as np, rasterio
from rasterio.warp import transform as wtransform
from scipy.ndimage import map_coordinates
R = 6371000.0
import os
_D = os.environ.get("DEM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "dem"))
DEMS = {"sierra": os.path.join(_D, "main_srtm90.tif"), "oregon": os.path.join(_D, "main_dem.tif")}
_cache = {}

def _dem(name):
    if name not in _cache:
        d = rasterio.open(DEMS[name]); a = d.read(1).astype(float)
        if d.nodata is not None: a[a == d.nodata] = np.nan
        _cache[name] = (d, a)
    return _cache[name]

def profile(name, lat0, lon0, bearing_deg, half=24500.0, dx=30.0):
    """Heights along a straight line through (lat0, lon0), bearing in degrees from north; x in [-half, half] m."""
    d, a = _dem(name)
    x = np.arange(-half, half + dx / 2, dx)
    b = np.radians(bearing_deg)
    lat = lat0 + np.degrees(x * np.cos(b) / R)
    lon = lon0 + np.degrees(x * np.sin(b) / (R * np.cos(np.radians(lat0))))
    if d.crs.to_epsg() != 4326:
        xs, ys = wtransform("EPSG:4326", d.crs, lon.tolist(), lat.tolist())
    else:
        xs, ys = lon, lat
    inv = ~d.transform
    col, row = inv * (np.asarray(xs), np.asarray(ys))
    h = map_coordinates(a, [np.asarray(row) - 0.5, np.asarray(col) - 0.5], order=1, mode="nearest")
    return x, h, lat, lon

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = {"Yosemite Valley, N-S": ("sierra", 37.735, -119.585, 0.0),
         "Owens Valley, W-E": ("sierra", 37.36, -118.47, 90.0),
         "Sierra foothills, W-E": ("sierra", 37.52, -120.30, 90.0),
         "Mono Lake basin, W-E": ("sierra", 37.95, -119.10, 90.0),
         "Oregon, W-E": ("oregon", 44.40, -122.72, 90.0),
         "Oregon, N-S": ("oregon", 44.40, -122.60, 0.0)}
    fig, axs = plt.subplots(len(C), 1, figsize=(10, 2.0 * len(C)))
    for ax, (k, (n, la, lo, br)) in zip(axs, C.items()):
        x, h, lat, lon = profile(n, la, lo, br)
        ax.plot(x / 1e3, h, color="#785A3C"); ax.fill_between(x / 1e3, h.min(), h, color="#EBE4D6")
        m = np.abs(x) <= 6000
        sl = np.degrees(np.arctan(np.abs(np.gradient(h, x)))).max()
        ax.set_title(f"{k}: {h.min():.0f}-{h.max():.0f} m, central ±6 km {h[m].min():.0f}-{h[m].max():.0f} m, max slope {sl:.0f} deg, nan {np.isnan(h).sum()}", fontsize=9)
        ax.axvspan(-6, 6, color="#1565C0", alpha=0.06)
    plt.tight_layout(); plt.savefig("profiles.png", dpi=70)
