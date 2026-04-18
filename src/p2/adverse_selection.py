"""Arrival intensities and adverse-selection extensions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class ArrivalModel:
    @staticmethod
    def intensity(delta: float | np.ndarray, A: float, kappa: float) -> float | np.ndarray:
        delta_arr = np.asarray(delta, dtype=float)
        values = A * np.exp(-kappa * np.maximum(delta_arr, 0.0))
        if values.ndim == 0:
            return float(values.item())
        return values

    @staticmethod
    def adverse_jump(fill_direction: int | str, epsilon: float) -> float:
        if isinstance(fill_direction, str):
            side = fill_direction.strip().lower()
            if side in {"ask", "sell", "-1"}:
                return float(epsilon)
            return float(-epsilon)
        if int(fill_direction) < 0:
            return float(epsilon)
        return float(-epsilon)


def calibrate_epsilon(lobster_data_path: str | Path) -> float:
    path = Path(lobster_data_path)
    if not path.exists():
        raise FileNotFoundError(f"LOBSTER data not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        return 0.0

    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return 0.0

    column = "price" if "price" in numeric.columns else numeric.columns[0]
    series = numeric[column].astype(float)
    if series.median() > 10_000:
        series = series / 10_000.0

    epsilon = float(series.diff().abs().dropna().mean() / 2.0) if len(series) > 1 else 0.0
    return epsilon
