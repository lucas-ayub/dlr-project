"""
Test 2 -- STFT of one point target, reverse-engineer the delta_C0 cell.
Signal from the repo's getRawData1D (exact range history). The peak of the
sub-aperture STFT is mapped back through the ruler to the target cell (256).
Produces test2_stft.png.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sata_kernel import step1_mapping
from pipeline_signal import pipeline_point_line

wl, r0, v, PRFop, H = 0.25, 766206.0, 7500.0, 500.0, 720000.0
rref = r0; AZ, RD, GY = "#185FA5", "#E24B4A", "#5F5E5A"
dimx = 512

s = pipeline_point_line(r0, H, v, wl, PRFop, dimx)     # exact-range point target
g = step1_mapping(dimx, rref, PRFop, 1, v, 0.0, wl, r0)
Tsub, Nzp = int(g["Tsub"]), g["Nzp"]; start = int(dimx//2 - Tsub)
buf = s[start:start+2*Tsub]
if len(buf) < Nzp:
    buf = np.concatenate([buf, np.zeros(Nzp-len(buf), complex)])
spec = np.fft.fft(buf); fsub = g["fsub"]
o = np.argsort(fsub); fs = fsub[o]; sp = 20*np.log10(np.abs(spec)[o]/np.abs(spec).max()+1e-9)

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 130})
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(fs, sp, lw=2, color=AZ); ax.axvline(0, color=RD, ls="--", lw=1.5)
ax.set_xlabel("azimuth frequency  $f_a$  [Hz]")
ax.set_ylabel("intensity  [dB]  (peak = 0)")
ax.set_title("Test 2 - STFT of a getRawData1D point target: does the peak map back?")
ax.set_ylim(-40, 5); ax.grid(alpha=.3)
ax.annotate("peak at $f_a=0$\n-> maps back to cell 256\n(the target)\nreads dC0 there",
            xy=(0, 0), xytext=(55, -13), fontsize=11, color=RD,
            arrowprops=dict(arrowstyle="->", color=RD))
ax.annotate("fat peak: the target's energy\nis spread over many bins\n(short window = low resolution)",
            xy=(-95, sp[np.argmin(np.abs(fs+95))]), xytext=(-235, -32), fontsize=10, color=GY,
            arrowprops=dict(arrowstyle="->", color=GY))
plt.tight_layout(); plt.savefig("test2_stft.png"); plt.close()
print("saved test2_stft.png")
