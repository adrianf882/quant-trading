"""Diferenciacion fraccionaria (Lopez de Prado, AFML cap. 5).

Dilema (linea 154, "The Stationarity vs. Memory Dilemma"): las series de precios
son no estacionarias, pero las transformaciones estandar (ej. diferenciar una
vez) borran la memoria (linea 2892: "standard stationarity transformations...
erase memory"). La diferenciacion fraccionaria (FFD) hace la serie estacionaria
conservando el maximo de memoria posible.

Se elige `d` como el menor valor que pasa el test ADF (p < 0.05).
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights(d: float, size: int) -> np.ndarray:
    """Pesos de la diferenciacion fraccionaria de orden d."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """Diferenciacion fraccionaria de ventana fija (FFD).

    Se descartan los pesos menores a `thres` para truncar la memoria infinita.
    """
    w = get_weights(d, len(series))
    w = w[np.abs(w) >= thres]
    width = len(w) - 1
    values = series.to_numpy(dtype=float)
    # y[i] = sum_k w[k] * x[i-k]  ->  convolucion (rapida, vectorizada).
    conv = np.convolve(values, w)
    out = conv[: len(values)].copy()
    out[:width] = np.nan  # primeras `width` sin historia completa
    return pd.Series(out, index=series.index)


def find_min_d(series: pd.Series, step: float = 0.1, thres: float = 1e-5) -> float:
    """Menor d (0..1) cuya serie FFD es estacionaria segun ADF (p < 0.05)."""
    for d in np.arange(0.0, 1.0 + step, step):
        fd = frac_diff_ffd(series, float(d), thres).dropna()
        if len(fd) < 100:
            continue
        p_value = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
        if p_value < 0.05:
            return float(d)
    return 1.0


def add_fracdiff_features(features: pd.DataFrame, df: pd.DataFrame,
                          d_close: float | None = None,
                          d_volume: float | None = None) -> pd.DataFrame:
    """Agrega columnas de precio y volumen diferenciados fraccionalmente."""
    log_close = np.log(df["close"])
    log_volume = np.log(df["volume"].replace(0, np.nan))

    if d_close is None:
        d_close = find_min_d(log_close)
    if d_volume is None:
        d_volume = find_min_d(log_volume.dropna())

    out = features.copy()
    out["ffd_close"] = frac_diff_ffd(log_close, d_close).reindex(features.index)
    out["ffd_volume"] = frac_diff_ffd(log_volume, d_volume).reindex(features.index)
    out.attrs["d_close"] = d_close
    out.attrs["d_volume"] = d_volume
    return out


if __name__ == "__main__":
    from .data import load_ohlcv, raw_data_path

    df = load_ohlcv(raw_data_path("BTC/USDT", "1d"))
    d = find_min_d(np.log(df["close"]))
    print(f"ADF sobre precio (sin diferenciar): p={adfuller(np.log(df['close']), maxlag=1, autolag=None)[1]:.3f}")
    print(f"d minimo que da estacionariedad: {d}")
    fd = frac_diff_ffd(np.log(df["close"]), d).dropna()
    print(f"ADF con FFD d={d}: p={adfuller(fd, maxlag=1, autolag=None)[1]:.4f}, "
          f"memoria preservada (n filas no-NaN={len(fd)})")
