"""Shared setup for the analyses in joint_height_dense (joint reconstruction, 1-D azimuth lines).

Same system, generators, focusing and metric as the results of the final presentation and of the report (chapter 5):
  default system N_rx = 4 (Rx0 = Tx), PRF 2000 Hz together, b_xt = 0/300/119.8/100.5 m, data from
  sata1d.generate_channels (exact distances, no noise), reference = one monostatic radar at the full PRF limited to
  the processed band B_a, standard = sata1d.reconstruct, joint = sata1d.joint_line.reconstruct_joint_line.

The only repository dependency is the sata1d package and two helper modules of sata1d_code/joint_cases
(audit_dem: the ramp terrain; run_joint_sweep / sweep_joint_all: the synthetic terrains). The repository root is
found from DLR_REPO or by walking up from this file until a folder with sata1d_code/ is found.
"""
import os, sys, time
from dataclasses import replace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo():
    r = os.environ.get("DLR_REPO")
    if r:
        return r
    d = HERE
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "sata1d_code")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repository root not found: set DLR_REPO")


REPO = _repo()
SC = os.path.join(REPO, "sata1d_code")
sys.path[:0] = [SC, os.path.join(SC, "joint_cases"), HERE]
import sata1d as S                                                      # noqa: E402
from sata1d.config import ArrayGeometry                                 # noqa: E402
from sata1d.joint_line import height_table, line_maps, reconstruct_joint_line  # noqa: E402

CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "fig")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

B300 = [0.0, 300.0, 119.8, 100.5]
B_ECHO = 1281.0          # Doppler band of one point echo [Hz] (as in compute_config.py)
HALF = 23600.0           # processed line about +-23.6 km (as in compute_real.py)


# ---------------------------------------------------------------- terrains
def gauss_mountain(H, sigma=1500.0):
    """The talk's isolated mountain (run_joint_sweep.TERRAINS['mountain']) with peak height H."""
    return lambda x: H * np.exp(-0.5 * (np.asarray(x, float) / sigma) ** 2)


def relative(h):
    """Terrain relative to its mean height over the processed line (reference height of the processing)."""
    x = np.linspace(-HALF, HALF, 9441)
    href = float(np.mean(h(x)))
    return (lambda q: np.asarray(h(np.asarray(q, float)), float) - href), href


def synth_terrain(name):
    """Synthetic terrains of the talk (compute_synth.TERRAINS)."""
    import run_joint_sweep as W
    import sweep_joint_all as SJ
    T = {"mountain": W.TERRAINS["mountain"][0], "mountains": W.TERRAINS["mountains"][0],
         "escarpment": W.TERRAINS["escarpment"][0], "plateau": W.TERRAINS["plateau_w"][0]}
    T.update({k: SJ.NEW_T[k][0] for k in ("alpine", "valley", "canyon", "volcano", "ridges", "rolling")})
    return T[name]


REAL = {   # name -> (DEM, centre lat, centre lon, bearing [deg from north], description) as compute_real/compute_v22
    "foothills": ("sierra", 37.52, -120.22, 90.0, "western Sierra foothills, west-east"),
    "highsierra": ("sierra", 37.80, -119.35, 0.0, "High Sierra near Tuolumne Meadows, north-south"),
    "escarpment": ("sierra", 37.36, -118.60, 90.0, "eastern Sierra escarpment into Owens Valley, west-east"),
    "yosemite": ("sierra", 37.722, -119.585, 0.0, "Yosemite Valley, north-south"),
}


def real_terrain(name):
    from profiles import profile
    dem, la, lo, br, _ = REAL[name]
    x, h, _, _ = profile(dem, la, lo, br, half=HALF + 900.0, dx=30.0)
    href = float(np.mean(h[np.abs(x) <= HALF]))
    return (lambda q: np.interp(np.asarray(q, float), x, h - href)), href


# ---------------------------------------------------------------- processing
def band(cfg, s):
    U = np.fft.fft(s)
    U[np.abs(np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)) > cfg.abw / 2] = 0
    return np.fft.ifft(U)


def focuser(cfg, tr):
    """Azimuth compression with the echo of the reference point (compute_real.focus); returns complex lines."""
    sref = S.generate_reference(cfg, tr, cfg.scene.ptg[None, :])
    Hf = np.conj(np.fft.fft(sref))
    return lambda s: np.roll(np.fft.ifft(np.fft.fft(s) * Hf), cfg.Na // 2)


def simulate(xs, terrain, amps=None, bxt=B300, prf=2000.0, methods=("ref", "std", "joint"), dem=None):
    """Run one 1-D line. Returns dict with complex focused lines (ref, std, joint), the axis and the geometry.
    dem: terrain given to the joint (default: the true terrain)."""
    xs = np.asarray(xs, float)
    hs = terrain(xs)
    cfg = S.make_config(list(zip(xs, hs)), nrx=len(bxt), prf=prf, bxt_max=0.0, amplitudes=amps, tx_rx_channel0=True)
    cfg = replace(cfg, array=ArrayGeometry(bat=cfg.array.bat, bxt=np.asarray(bxt, float).copy()))
    tr = S.build_platform_tracks(cfg)
    F = focuser(cfg, tr)
    out = dict(xs=xs, hs=np.asarray(hs, float), Na=cfg.Na, prf=prf, prf_op=cfg.PRF_op, vs=cfg.system.vs,
               t=(np.arange(cfg.Na) - cfg.Na // 2) / cfg.prf, abw=cfg.abw, sec={})
    out["dt_amb"] = cfg.PRF_op * cfg.system.wl * cfg.scene.r0 / (2.0 * cfg.system.vs ** 2)
    out["X"] = out["dt_amb"] * cfg.system.vs
    t0 = time.time()
    out["ref"] = F(band(cfg, S.generate_reference(cfg, tr)))
    out["sec"]["ref"] = time.time() - t0
    if "std" in methods or "joint" in methods:
        t0 = time.time()
        ch = S.generate_channels(cfg, tr)
        out["sec"]["data"] = time.time() - t0
    if "std" in methods:
        t0 = time.time()
        out["std"] = F(S.reconstruct(cfg, tr, ch))
        out["sec"]["std"] = time.time() - t0
    if "joint" in methods:
        t0 = time.time()
        half = 0.5 * cfg.Na / cfg.prf * cfg.system.vs
        d = terrain if dem is None else dem
        hgrid = d(np.linspace(-half - 100.0, half + 100.0, 8001))
        tab = height_table(cfg, tr, hgrid.min() - 60.0, hgrid.max() + 60.0)
        y, info = reconstruct_joint_line(cfg, tr, ch, line_maps(cfg, tr, d, tab), return_info=True)
        out["joint"] = F(y)
        out["iterations"] = info.get("iterations")
        out["sec"]["joint"] = time.time() - t0
    return out


# ---------------------------------------------------------------- metrics
def amb_old(r, key, win=0.02, guard=0.02):
    """The talk's metric (audit_dem.ambiguity_level, as used for the results of the report): largest |line| within +-win s of every
    ambiguity position x_p/v + m dt_amb (0 < |m| <= floor(1281 Hz / PRF_op)) of every target p, leaving out +-guard s
    around every target, in dB re. the reference peak."""
    t, env, pk = r["t"], np.abs(r[key]), np.abs(r["ref"]).max()
    M = int(B_ECHO // r["prf_op"])
    lobe = np.zeros(t.size, bool)
    sel = np.zeros(t.size, bool)
    for x in r["xs"]:
        lobe |= np.abs(t - x / r["vs"]) <= guard
        for m in range(-M, M + 1):
            if m:
                sel |= np.abs(t - (x / r["vs"] + m * r["dt_amb"])) <= win
    sel &= ~lobe
    return 20 * np.log10(env[sel].max() / pk + 1e-15) if sel.any() else np.nan


def ambiguity_positions(r):
    """Where each ambiguity of each target is (m = +-1, +-2), from the peaks of the standard reconstruction of
    the flat scene is not needed: dt_amb follows from the geometry (all targets on the iso-range of the reference)."""
    M = int(B_ECHO // r["prf_op"])
    return np.array([x / r["vs"] + m * r["dt_amb"] for x in r["xs"] for m in range(-M, M + 1) if m])


def amb_new(r, key, win=0.02, guard=0.02):
    """Checked metric (see README). Returns (amb, floor, peak_loss):
       amb       = the talk's metric, recomputed here (largest level within +-win of every predicted ambiguity
                   position, outside +-guard around every true target), dB re. the reference peak;
       floor     = the largest level ANYWHERE on the line outside +-guard around every true target (and away from
                   the circular wrap at the line ends): catches artefacts that are not at the predicted positions;
       peak_loss = the weakest true-target peak, dB re. the reference peak (energy that leaves the targets)."""
    t, env, pk = r["t"], np.abs(r[key]), np.abs(r["ref"]).max()
    lobe = np.zeros(t.size, bool)
    for x in r["xs"]:
        lobe |= np.abs(t - x / r["vs"]) <= guard
    sel = np.zeros(t.size, bool)
    for tp in ambiguity_positions(r):
        sel |= np.abs(t - tp) <= win
    a = 20 * np.log10(env[sel & ~lobe].max() / pk + 1e-15)
    rng = np.abs(t) <= (np.abs(r["xs"]).max() + 2.2 * r["X"]) / r["vs"]
    b = 20 * np.log10(env[~lobe & rng].max() / pk + 1e-15)
    pk_up = width3db(r, "ref", r["xs"][int(np.argmax([width3db(r, "ref", x)[1] for x in r["xs"]]))])[1]
    c = min(width3db(r, key, x)[1] for x in r["xs"]) - pk_up    # true peaks, upsampled, re. the reference peak
    return float(a), float(b), float(c)


def mainlobe_halfwidth(r):
    """Half width [s] of the reference main lobe from peak to first null (searched around the strongest target)."""
    env = np.abs(r["ref"])
    i = int(np.argmax(env))
    j = i
    while j + 1 < env.size and env[j + 1] < env[j]:
        j += 1
    k = i
    while k - 1 >= 0 and env[k - 1] < env[k]:
        k -= 1
    return 0.5 * (j - k) * (r["t"][1] - r["t"][0]) * 1.0


def width3db(r, key, x, os=16):
    """3-dB width [m] of the main lobe of |line| at target x (upsampled by zero padding in frequency)."""
    t, line = r["t"], r[key]
    i = int(np.argmin(np.abs(t - x / r["vs"])))
    n = 64
    seg = line[i - n:i + n]
    U = np.fft.fftshift(np.fft.fft(seg))
    Up = np.zeros(seg.size * os, complex)
    c = Up.size // 2
    Up[c - seg.size // 2:c + seg.size // 2] = U
    s = np.abs(np.fft.ifft(np.fft.ifftshift(Up)))
    k = int(np.argmax(s))
    half = s[k] / np.sqrt(2)
    a = k
    while a > 0 and s[a] > half:
        a -= 1
    b = k
    while b < s.size - 1 and s[b] > half:
        b += 1
    dt = (t[1] - t[0]) / os
    return (b - a) * dt * r["vs"], 20 * np.log10(s[k] * os / np.abs(r["ref"]).max() + 1e-15)


def nmse_db(r, key):
    """Normalised error of the focused complex line vs the reference (same band): 20 log10 ||y - ref|| / ||ref||."""
    y, ref = r[key], r["ref"]
    return float(20 * np.log10(np.linalg.norm(y - ref) / np.linalg.norm(ref)))
