"""Tiny SYNTHETIC stand-in with the AI4I column layout - FOR TESTS ONLY.

It is NOT the real dataset and must never be used to report results. It only
lets unit tests run without downloading the real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_raw(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    typ = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    speed = rng.normal(1540, 170, n).clip(1100, 2900)
    torque = rng.normal(40, 10, n).clip(3, 75)
    wear = rng.integers(0, 250, n).astype(float)
    twf = (wear > 225) & (rng.random(n) < 0.4)
    hdf = ((proc - air) < 8.6) & (speed < 1380)
    pwf = (torque * speed * 2 * np.pi / 60 < 3500) | (torque * speed * 2 * np.pi / 60 > 9000)
    osf = wear * torque > 11000
    rnf = rng.random(n) < 0.001
    fail = (twf | hdf | pwf | osf | rnf).astype(int)
    return pd.DataFrame({
        "UID": np.arange(1, n + 1), "Product ID": [f"{t}{i}" for i, t in enumerate(typ)],
        "Type": typ, "Air temperature [K]": air.round(1), "Process temperature [K]": proc.round(1),
        "Rotational speed [rpm]": speed.round(0), "Torque [Nm]": torque.round(1),
        "Tool wear [min]": wear, "Machine failure": fail,
        "TWF": twf.astype(int), "HDF": hdf.astype(int), "PWF": pwf.astype(int),
        "OSF": osf.astype(int), "RNF": rnf.astype(int),
    })
