"""run_terrain_lut_cases.py with a single channel (Nrx = 1, PRF_op = PRF = 2000 Hz: no folds), same CLI.
--mono also puts the channel on the transmitter (tx_rx_channel0: bat = bxt = 0, monostatic).
    python run_terrain_lut_nrx1.py --cases COS-B80-A80-P20 --n-targets 7 --out sweep_terrain_rerun_nrx1
    python run_terrain_lut_nrx1.py --mono --cases COS-B80-A80-P20 --n-targets 7 --out sweep_terrain_rerun_nrx1_mono
"""
import functools
import sys
import sata1d as S
MONO = "--mono" in sys.argv
if MONO:
    sys.argv.remove("--mono")
S.make_config = functools.partial(S.make_config, nrx=1, tx_rx_channel0=MONO)
import run_terrain_lut_cases as T
if __name__ == "__main__":
    T.main()
