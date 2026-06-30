# SAR Multichannel Reconstruction Pipeline — Customization Guide

This document explains how to configure and extend the modular SAR
multichannel azimuth reconstruction pipeline (`sar_recon/`), without having
to touch the signal model or the reconstruction math.

## Package layout

```
run_experiment.py        <- driver script (sweeps Nrx x case x scene)
sar_recon/
    config.py             <- SystemParams, Scene, ArrayGeometry, ExperimentConfig,
                             SCENE_PRESETS, CONFIG_FACTORIES  <-- you mostly edit here
    geometry.py            <- builds TX/RX tracks from a config (no need to touch)
    signal_model.py         <- getRawData1D + reference/channel generation (verbatim core)
    reconstruction.py        <- GetCoeffNu / Hf / ReconstructSignalNumeri (verbatim core)
    analysis.py                <- matched filter + IRF zoom (no need to touch)
    plotting.py                  <- all plot_* functions (no need to touch)
```

Everything that varies between runs — system physics, array geometry, PRF
strategy, scene/target layout — lives in `config.py` as plain dataclasses.
The rest of the pipeline (`geometry.py`, `signal_model.py`,
`reconstruction.py`, `analysis.py`, `plotting.py`) only consumes an
`ExperimentConfig` object and never needs to change when you add a new
scenario.

## The building blocks

| Dataclass         | What it controls                                                   |
|--------------------|---------------------------------------------------------------------|
| `SystemParams`     | Radar/platform physics: `wl`, `ve`, `vs`, antenna length factors. `abw` and `theta` are derived automatically. |
| `Scene`            | Target range delay (`rDelay` -> `r0` -> `y0`), central point `(x0, y0, h0)`, and `extra_offsets` for additional scatterers. |
| `ArrayGeometry`    | Receiver baselines: `bat` (along-track) and `bxt` (cross-track). Use `ArrayGeometry.linear(Nrx, dx, dxt)` for a uniform array, or pass arbitrary arrays directly. |
| `ExperimentConfig` | Bundles everything above plus `prf`, `PRF_op`, the time axis (`Na`, `Na_ch`, `ta`), and `plots_dir`. This is the single object every pipeline function takes as input. |

There are three independent things you can customize, ordered from simplest
to most involved.

---

## Level 1 — Add a new scene (a set of scattering points)

This is the simplest change and the one you'll use most often. It does
**not** require touching anything other than `config.py`.

In `sar_recon/config.py`, `SCENE_PRESETS` is a dictionary of named scenes.
Each entry is a tuple of `(dx, dy, dh)` offsets in metres, **relative to the
central reconstruction point**:

```python
SCENE_PRESETS = {
    "single": (),                       # original one-target behaviour
    "my_scene": (
        (10.0, 5.0, 0.0),
        (-10.0, -5.0, 2.0),
    ),
    # add as many named scenes as you like
}
```

The central point itself is always included automatically — you only list
the *extra* scatterers. Signal generation (`generate_reference`,
`generate_channels`) sums the contribution of every point in the scene, but
reconstruction always uses the single central point (`cfg.scene.ptg`), so no
other code needs to know how many scatterers were used to generate the
signal.

To run it, add the name to `SCENE_NAMES` in `run_experiment.py`:

```python
SCENE_NAMES = ["single", "my_scene"]
```

> **Important:** `SCENE_NAMES` defaults to `["single"]`. Adding an entry to
> `SCENE_PRESETS` does **not** run it automatically — you must also list its
> name in `SCENE_NAMES`, or the driver only ever generates the original
> single-target output (no `scene_points` / `scene_points_3d` folders will
> appear at all).

The driver then sweeps every combination of `Nrx x case x scene_name`
automatically, and `run_case()` calls the scene-layout plots
(`plot_scene_points`, `plot_scene_points_3d`) whenever the scene has extra
offsets.

**Why offsets, not absolute coordinates?** Because the central point's
absolute position depends on `rDelay` (and therefore on `r0`/`y0`, derived
per case), offsets keep the scene definition independent of which base case
(`diff`, `dpca`, or a custom one) it's combined with.

---

## Setting the central point

The central point is `(x0, y0, h0)`. `x0` (along-track) and `h0` (height)
are plain fields you set directly. `y0` (the ground/cross-track position) is
**not** a field — it's derived from `rDelay`:

```python
r0 = c0 * rDelay / 2.0
y0 = sqrt(r0**2 - (H - h0)**2)
```

This reflects the actual radar geometry: you don't pick a ground position
directly, you pick the round-trip echo delay, which is what determines the
viewing geometry.

You can build a `Scene` either way:

```python
# Original way: think in terms of range delay.
scene = sar.Scene(rDelay=0.0051115753, x0=20.0, h0=2.0)

# Convenience way: think in terms of the target position directly.
# rDelay is back-computed so that scene.y0 == y0 exactly.
scene = sar.Scene.from_target(x0=20.0, y0=262057.8, h0=2.0)
```

`Scene.from_target(x0, y0, h0, H=720e3, c0=..., extra_offsets=())` solves
`r0 = sqrt(y0**2 + (H - h0)**2)` and sets `rDelay = 2*r0/c0` for you. Use
this whenever it's easier to reason about "I want my target here" rather
than "I want this echo delay". Note that `y0` depends on the platform height
`H` (the slant range / ground range / height form a right triangle), so
changing `H` in a custom case changes the `y0` you get for a fixed `rDelay`
— but `Scene.from_target` always gives you back the exact `y0` you asked
for, regardless of `H`.

To use it inside a `make_*_config` factory, just swap the `Scene(...)` call:

```python
scene = Scene.from_target(x0=20.0, y0=262057.8, h0=2.0, c0=system.c0,
                          extra_offsets=SCENE_PRESETS[scene_name])
```

---

## Level 2 — Tweak parameters of an existing case

If you just want to change numbers inside `make_diff_config` or
`make_dpca_config` (system physics, baseline spacing, PRF value, range
delay), edit them directly — there's no indirection to worry about:

```python
def make_diff_config(Nrx: int, base_dir: str, scene_name: str = "single") -> ExperimentConfig:
    system = SystemParams()                     # <- wl, ve, vs, antenna factors
    scene = Scene(rDelay=0.0051115753, ...)      # <- target range delay
    dx, dxt = 100.0, 200.0                       # <- array spacing
    array = ArrayGeometry.linear(Nrx, dx, dxt)
    prf, PRF_op = prf_from_fixed(2000.0, Nrx)    # <- PRF strategy
    ...
```

Everything downstream (`Na`, `ta`, `abw`, `theta`, the platform tracks, the
signal, the reconstruction) is derived from these few inputs, so changing
them here is enough.

---

## Level 3 — Create a brand-new case (different system / array / PRF strategy)

When you need a system, array geometry, or PRF strategy that's genuinely
different from `diff` and `dpca` (rather than just different numbers), write
a new factory function following the same shape:

```python
def make_my_case_config(Nrx: int, base_dir: str, scene_name: str = "single") -> ExperimentConfig:
    # 1) System physics
    system = SystemParams(wl=0.20, ve=7000.0, vs=7200.0)

    # 2) Scene: range delay + any extra scatterers from SCENE_PRESETS
    scene = Scene(rDelay=0.0045, c0=system.c0,
                  extra_offsets=SCENE_PRESETS[scene_name])
    # or, equivalently: Scene.from_target(x0=..., y0=..., h0=..., c0=system.c0,
    #                                      extra_offsets=SCENE_PRESETS[scene_name])

    # 3) Receiver array
    array = ArrayGeometry.linear(Nrx, dx=50.0, dxt=150.0)

    # 4) PRF strategy: fixed PRF or DPCA spacing
    prf, PRF_op = prf_from_fixed(1800.0, Nrx)
    # or:  prf, PRF_op = prf_from_dpca(system, Nrx, dx)

    # 5) Time axis (always derived the same way, no need to change)
    acq_time = 2.0 * integration_time(system, scene)
    Na, Na_ch, ta = build_time_axis(prf, Nrx, acq_time)

    # 6) Output folder, organized per scene (same convention as the existing cases)
    plots_dir = _plots_subdir(os.path.join(base_dir, "plots_my_case"), scene_name)

    return ExperimentConfig(
        name="my_case", system=system, scene=scene, array=array,
        prf=prf, PRF_op=PRF_op, Na=Na, Na_ch=Na_ch, ta=ta, plots_dir=plots_dir,
    )
```

Register it next to the existing cases:

```python
CONFIG_FACTORIES = {
    "diff": make_diff_config,
    "dpca": make_dpca_config,
    "my_case": make_my_case_config,
}
```

And add the name to `run_experiment.py`:

```python
CASES = ["diff", "dpca", "my_case"]
```

Nothing else needs to change — the driver, the signal model, the
reconstruction, and the plotting code only ever see an `ExperimentConfig`.

---

## Running a one-off configuration (no sweep)

For quick experiments — a notebook cell, a scratch script — you don't need
to register anything in `CONFIG_FACTORIES` or edit `run_experiment.py` at
all. Just call the pipeline functions directly with whatever `cfg` you build
(or grab one from a factory and tweak it):

```python
import sar_recon as sar

cfg = sar.CONFIG_FACTORIES["diff"](Nrx=5, base_dir=".", scene_name="varied_heights")
tracks = sar.build_platform_tracks(cfg)

sref = sar.generate_reference(cfg, tracks)
s_channel = sar.generate_channels(cfg, tracks)
srecN = sar.reconstruct(cfg, tracks, s_channel)

res = sar.analyze(cfg, sref, srecN)
sar.plot_combined(cfg, res)
sar.plot_scene_points_3d(cfg)
```

This is exactly the body of `run_case()` in `run_experiment.py` — useful
when you want to iterate fast on a single `Nrx`/scenario instead of running
the full sweep.

---

## Quick reference

| You want to...                                              | Edit...                                                  |
|----------------------------------------------------------------|-----------------------------------------------------------|
| Add more scattering points to a scene                          | `SCENE_PRESETS` in `config.py`                            |
| Actually run a scene you added to `SCENE_PRESETS`              | `SCENE_NAMES` in `run_experiment.py` (defaults to `["single"]` only) |
| Set the central point by position instead of range delay       | `Scene.from_target(x0=..., y0=..., h0=...)` instead of `Scene(rDelay=...)` |
| Change wavelength, velocities, or antenna size                 | `SystemParams(...)` inside the relevant `make_*_config`   |
| Change receiver spacing / array layout                         | `ArrayGeometry.linear(...)` (or build `bat`/`bxt` by hand) |
| Change PRF / PRF strategy                                      | `prf_from_fixed(...)` or `prf_from_dpca(...)` call         |
| Change target range / height                                   | `Scene(rDelay=..., h0=...)` or `Scene.from_target(y0=..., h0=...)` |
| Add a whole new system/array/PRF combination                   | New `make_*_config` function + entry in `CONFIG_FACTORIES` |
| Run a new combination as part of the sweep                     | `CASES` / `SCENE_NAMES` in `run_experiment.py`               |
| Generate plots showing where the scatterers are                | `plot_scene_points(cfg)` (2D, exact) and `plot_scene_points_3d(cfg)` (3D, zoomed) — both auto-called by `run_case()` when `cfg.scene.extra_offsets` is non-empty |

## Plot outputs

Every `make_*_config` organizes its plots as:

```
plots_dir = <base plots folder>/[<scene_name>/]
    combined/plot_combined_NrxN.png
    polyfit/plot_polyfit_NrxN_CHk.png
    geometry_3d/plot_geometry_3d_NrxN.png
    scene_points/plot_scene_points_NrxN.png          (only if scene has extra points)
    scene_points_3d/plot_scene_points_3d_NrxN.png    (only if scene has extra points)
```

The `<scene_name>/` subfolder is omitted for the default `"single"` scene,
so the original (single-target) output layout is unchanged. This also means
that if you only ever see `combined/`, `geometry_3d/`, and `polyfit/` with no
`<scene_name>/` subfolder and no `scene_points*` folders, it's a sign that
`SCENE_NAMES` in `run_experiment.py` is still set to `["single"]` only — see
the callout in Level 1 above.

- `geometry_3d` shows the full acquisition geometry (TX/RX tracks + scene) at
  kilometre scale — useful for context, but scatterers a few metres apart are
  indistinguishable there because slant range is typically hundreds of km.
- `scene_points` / `scene_points_3d` zoom into the scene itself, in metres,
  relative to the central reconstruction point — use these to actually
  inspect the layout of multiple scatterers.
