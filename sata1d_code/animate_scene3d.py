"""
GIF of an acquisition in 3-D: platform motion, Nrx receivers, topography, highlighted targets
(sata1d.scene3d.animate_scene).

    python animate_scene3d.py                                           # b_at = 0, b_xt = [63.7, 27.0, 4.1, 1.7]
    python animate_scene3d.py --bat 0 7.69 15.38 23.07 --bxt 0 0 0 0     # DPCA along-track array
    python animate_scene3d.py --nrx 6 --bat-step 7.69 --bxt-max 100     # 6 channels, DPCA, b_xt ~ U(0, 100)
    python animate_scene3d.py --terrain "800*exp(-0.5*((x/2000)**2+(y/1500)**2))" --targets=-3000:-1500,0:0,2500:500:300

terrain : numpy expression in x (azimuth) and y (ground range) [m] relative to the reference point,
          or one of the presets: hill (default), ramp, ridge, flat
targets : x:y (height from the terrain) or x:y:h, comma-separated
"""
import argparse
import numpy as np
from sata1d.scene3d import animate_scene

PRESETS = {
    "hill": "900*exp(-0.5*(((x-1000)/1800)**2+((y-1000)/1500)**2))"
            " + 400*exp(-0.5*((x+2500)/900)**2)*exp(-0.5*((y+800)/2500)**2) + 0.05*(y+4000)",
    "ramp": "1000*(x+6000)/12000",
    "ridge": "600*exp(-0.5*(y/800)**2)",
    "flat": "0*x",
}


def terrain_from(expr):
    expr = PRESETS.get(expr, expr)
    ns = {k: getattr(np, k) for k in ("exp", "sin", "cos", "tan", "sqrt", "abs", "pi", "tanh", "arctan", "clip", "maximum", "minimum")}
    return lambda x, y: np.broadcast_to(eval(expr, {"__builtins__": {}}, dict(ns, x=x, y=y)), np.shape(x)) * 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat", type=float, nargs="*", help="along-track baselines [m], one per channel")
    ap.add_argument("--bxt", type=float, nargs="*", help="cross-track baselines [m], one per channel")
    ap.add_argument("--nrx", type=int, default=4, help="channels when --bat/--bxt are not given")
    ap.add_argument("--bat-step", type=float, default=0.0, help="b_at = step * i when --bat is not given")
    ap.add_argument("--bxt-max", type=float, default=100.0, help="b_xt ~ U(0, bxt_max) when --bxt is not given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain", default="hill")
    ap.add_argument("--targets", default="-3000:-1500,-1500:500,0:0,1000:1500,2500:-500,4000:2500")
    ap.add_argument("--extent", type=float, nargs=2, default=(6000.0, 4000.0), help="half-sizes x y of the terrain [m]")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=4, help="frames per second of the GIF (lower = slower)")
    ap.add_argument("--channels", type=int, nargs="*", default=None, help="channels in the range-history panel")
    ap.add_argument("--range-mode", default="min", choices=["min", "offset"],
                    help="min: each range history minus its minimum [m]; offset: R_T + R_R - 2 r0 [km]")
    ap.add_argument("--t-span", type=float, nargs=2, default=None, help="start and end time of the animation [s]")
    ap.add_argument("--out", default="scene3d.gif")
    a = ap.parse_args()

    nrx = len(a.bat) if a.bat else (len(a.bxt) if a.bxt else a.nrx)
    bat = np.array(a.bat) if a.bat else a.bat_step * np.arange(nrx)
    bxt = np.array(a.bxt) if a.bxt else np.random.default_rng(a.seed).uniform(0.0, a.bxt_max, nrx)
    targets = [tuple(float(v) for v in s.split(":")) for s in a.targets.split(",") if s]
    out = animate_scene(bat, bxt, terrain_from(a.terrain), targets, out=a.out, extent=tuple(a.extent),
                        frames=a.frames, fps=a.fps, channel_ranges=a.channels,
                        range_mode=a.range_mode, t_span=tuple(a.t_span) if a.t_span else None)
    print(f"Nrx = {nrx}, b_at = {np.round(bat, 2)} m, b_xt = {np.round(bxt, 2)} m -> {out}")


if __name__ == "__main__":
    main()
