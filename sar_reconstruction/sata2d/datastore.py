# -*- coding: utf-8 -*-
"""
Tiny stand-in for the ``h5py.File`` object used by the reference code.

The reference two-step reconstruction is written *out-of-core*: it opens an
HDF5 file and addresses the big arrays as ``ds['CH'][...]``, ``ds['REC'][...]``,
``ds['REC_Pref'][...]``.  That is essential at the "paper" preset (tens of TB)
but is pure overhead at the debugging preset, and it makes the package depend
on h5py.

``DataStore`` gives the same ``ds['NAME'][...]`` read/write syntax on top of
either

* plain NumPy arrays in RAM (``backend="memory"``, the default), or
* a real HDF5 file (``backend="hdf5"``) if ``h5py`` is installed.

So the reconstruction code below is *literally* the reference code, unchanged,
and you can flip one flag to go from a fast in-RAM debug run to an out-of-core
production run.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DataStore"]


class DataStore:
    def __init__(self, filename: str | None = None, backend: str = "memory"):
        self.backend = backend
        self._arrays: dict[str, np.ndarray] = {}
        self._h5 = None
        if backend == "hdf5":
            import h5py  # imported lazily: optional dependency
            self._h5 = h5py.File(filename, "w", libver="latest")
        elif backend != "memory":
            raise ValueError("backend must be 'memory' or 'hdf5'")

    # -- creation ----------------------------------------------------------
    def create(self, name: str, shape, dtype=np.complex64) -> None:
        if self._h5 is not None:
            self._h5.create_dataset(name, shape=shape, dtype=dtype)
        else:
            self._arrays[name] = np.zeros(shape, dtype)

    # -- h5py-like access --------------------------------------------------
    def __getitem__(self, name: str):
        return self._h5[name] if self._h5 is not None else self._arrays[name]

    def __contains__(self, name: str) -> bool:
        return name in (self._h5 if self._h5 is not None else self._arrays)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
