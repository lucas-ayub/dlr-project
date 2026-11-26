# Animation of the joint reconstruction

`make_gif.py` writes `report/figures/readme_animation.gif`, the animation on the repository's front page. It uses only
the package `sata1d` (same generators, reconstruction and metric as the report):

- scene: Gaussian mountain (1000 m, sigma 1.5 km), five targets at x = -8, -0.55, 0.65, 2.9, 4.1 km on the surface,
  heights relative to the mean height of the line;
- system: the default formation, N_rx = 4 (Rx0 = Tx), 500 Hz per channel, cross-track baselines 0 / 300 / 119.8 /
  100.5 m, lambda = 0.25 m, r0 = 766 km.

Frames:
1. the formation flies along the line; its beam footprint (L_sa = lambda r0 / L_a, about 16 km) lights the targets it
   covers, and each receiver's channel (Re s_i at 500 Hz) is drawn up to the current position;
2. the focused line of the reference (one radar at the full PRF);
3. the standard reconstruction (flat-earth filters);
4. the joint's first estimate A^H s (each channel back on the places with its DEM phase, then summed; peak-normalised);
5. the joint reconstruction after the regularised iterations.

The highest ambiguity is printed and shown for each line (reference -41.4 dB, standard -4.4 dB, first estimate
-4.9 dB, joint -40.8 dB, re. the reference peak).

```bash
python sata1d_code/joint_cases/animation/make_gif.py            # ~30 s, about 2.5 MB
python sata1d_code/joint_cases/animation/make_gif.py --mp4      # also an .mp4 next to the GIF (needs ffmpeg)
python sata1d_code/joint_cases/animation/make_gif.py --out demo.gif --fps 10
```
