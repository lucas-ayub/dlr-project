"""Checks that the standalone copy (joint1d/) gives the same numbers as the project package (../sata1d)."""
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import joint1d as J
import sata1d as S

terrain = lambda x: 1000.0 * np.exp(-0.5 * (np.asarray(x, float) / 1500.0) ** 2)
XS = np.array([-5200.0, -2600.0, 0.0, 2600.0, 5200.0])
BXT = np.array([0.0, 134.9, 20.5, 8.3])
out = {}
for name, P in (("standalone", J), ("package", S)):
    cfg = P.make_config(list(zip(XS, terrain(XS))), nrx=4, bxt_max=0.0, tx_rx_channel0=True)
    cfg = replace(cfg, array=type(cfg.array)(bat=cfg.array.bat, bxt=BXT))
    tr = P.build_platform_tracks(cfg)
    ch = P.generate_channels(cfg, tr)
    out[name] = P.reconstruct_joint_line(cfg, tr, ch, P.line_maps(cfg, tr, terrain))
a, b = out["standalone"], out["package"]
print("max |standalone - package| / max |package| =", float(np.abs(a - b).max() / np.abs(b).max()))
