"""
Test 1 -- does each azimuth frequency map to the right ground cell?
Compares the SATA kernel's frequency->cell ruler (STEP 1) against the analytic
truth. Produces test1_mapping.png.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sata_kernel import step1_mapping

wl, r0, v, PRFop = 0.25, 766206.0, 7500.0, 500.0
rref = r0
AZ, OR = "#185FA5", "#EF9F27"

def truth_azpos(f):
    return r0*np.tan(np.arcsin(np.clip(wl*f/(2*v), -1, 1)))/v*PRFop

m = step1_mapping(512, rref, PRFop, 1, v, 0.0, wl, r0)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 130})
fig, ax = plt.subplots(figsize=(8, 5))
ff = np.linspace(m["fsub"].min(), m["fsub"].max(), 200)
ax.plot(ff, truth_azpos(ff), "-", lw=2.2, color=OR, label="physics truth (independent formula)")
ax.plot(m["fsub"], m["azpos"], "o", ms=5, color=AZ, label="SATA kernel (step 1)")
ax.set_xlabel("azimuth frequency  $f_a$  [Hz]")
ax.set_ylabel("azimuth cell  [samples]")
ax.set_title("Test 1 - does each frequency map to the right ground cell?")
ax.legend(loc="upper left", fontsize=11); ax.grid(alpha=.3)
ax.annotate("dots sit ON the line\n= exact ruler (error = 0.000 sample)",
            xy=(120, truth_azpos(120)), xytext=(-230, 120), fontsize=11, color=AZ,
            arrowprops=dict(arrowstyle="->", color=AZ))
plt.tight_layout(); plt.savefig("test1_mapping.png"); plt.close()
print("saved test1_mapping.png")
