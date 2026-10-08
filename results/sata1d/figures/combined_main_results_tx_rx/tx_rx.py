"""
Channel 0 on the transmitter (b_at = b_xt = 0) for every configuration the sata1d_code scripts build:
make_config(..., tx_rx_channel0=True). The other channels keep their b_xt (27.0, 4.1 and 1.7 m with seed 0).

    python tx_rx.py <script of sata1d_code> [its arguments]     # runs that script with the transmitter channel

(PYTHONPATH must contain sata1d_code.)
"""
import functools
import os
import runpy
import sys

import sata1d as S
import sata1d.config as C


def enable():
    if getattr(S.make_config, "tx_rx", False):
        return
    f = functools.partial(C.make_config, tx_rx_channel0=True)
    f.tx_rx = True
    S.make_config = f


if __name__ == "__main__":
    enable()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
    runpy.run_path(sys.argv[0], run_name="__main__")
