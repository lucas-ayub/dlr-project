"""Height change within one DEM sample at the targets: the joint reconstruction has one height per line sample
(dx = v / PRF = 3.84 m); a target between two samples whose heights differ by dh_s gets a phase between the two.
dh_s = max over targets of |h(x_k+1) - h(x_k)| for the two samples around the target; phase 2 pi b_xt dh_s /
(lambda r0 tan theta) for the longest baseline (300 m)."""
import numpy as np
DX = 3.8442685321606405
WL, R0, TH, B = 0.25, 766206.0, np.arccos(720e3 / 766206.0), 300.0


def terrain_abs(case, dh, xs0, w):
    if case in ("step", "three", "width"):
        return lambda x: 0.5 * dh * (1.0 + np.tanh(2.0 * (np.asarray(x, float) - xs0) / w))
    return lambda x: dh * np.exp(-0.5 * ((np.asarray(x, float) - xs0) / 10.0) ** 2)


def dh_sample(row):
    h = terrain_abs(row["case"], row["dh"], 0.5 * DX, row["w"])
    out = 0.0
    for x in row["xs"]:
        k = np.floor(x / DX)
        out = max(out, abs(float(h((k + 1) * DX) - h(k * DX))))
    return out


def phase(dhs):
    return 2 * np.pi * B * dhs / (WL * R0 * np.tan(TH))
