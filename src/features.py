"""Ingenieria de features causales.

Regla de oro: toda feature en el instante `t` se calcula SOLO con datos hasta
`t` (inclusive). Ninguna funcion de este modulo puede mirar hacia adelante.

Se usan transformaciones rolling/ewm, que por construccion solo dependen del
pasado. El test `tests/test_features.py` lo verifica de forma automatica con el
metodo de truncado que recomienda Chan (Quantitative Trading, cap. 3):

    "Run the program using all your historical data... Now truncate your
     historical data... check if [the results] are identical."

Uso:
    py -m src.features    # construye y guarda data/processed/features.parquet
"""
import numpy as np
import pandas as pd

from .config import load_config
from .data import load_ohlcv, processed_path, raw_data_path


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder. Usa suavizado exponencial (solo pasado)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Devuelve (macd, signal, histograma). EMAs causales."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def build_features(df: pd.DataFrame, vol_window: int = 24) -> pd.DataFrame:
    """Construye la matriz de features. Devuelve un DataFrame solo de features."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    log_close = np.log(close)
    feats = pd.DataFrame(index=df.index)

    # Momentum: retornos logaritmicos a 1, 4 y 24 velas.
    feats["ret_1"] = log_close.diff(1)
    feats["ret_4"] = log_close.diff(4)
    feats["ret_24"] = log_close.diff(24)

    # Volatilidad realizada (desvio de retornos de 1 vela).
    feats["vol_24"] = log_close.diff().rolling(vol_window).std()

    # Indicadores de fuerza/trend.
    feats["rsi_14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    feats["macd"] = macd_line
    feats["macd_signal"] = signal_line
    feats["macd_hist"] = hist

    # Volumen relativo a su promedio reciente.
    feats["vol_rel"] = volume / volume.rolling(vol_window).mean()

    # Distancia (relativa) a medias moviles simples.
    for w in (20, 50, 200):
        feats[f"dist_sma{w}"] = close / close.rolling(w).mean() - 1

    # Forma de la vela.
    feats["range_pct"] = (high - low) / close
    feats["close_pos"] = (close - low) / (high - low).replace(0, np.nan)

    return feats


FEATURE_COLUMNS = [
    "ret_1", "ret_4", "ret_24", "vol_24", "rsi_14",
    "macd", "macd_signal", "macd_hist", "vol_rel",
    "dist_sma20", "dist_sma50", "dist_sma200", "range_pct", "close_pos",
]


def main() -> None:
    data_cfg = load_config()["data"]
    symbol, timeframe = data_cfg["symbol"], data_cfg["timeframe"]
    df = load_ohlcv(raw_data_path(symbol, timeframe))
    feats = build_features(df)

    out_path = processed_path(symbol, timeframe, "features")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path)

    print(f"Features: {len(FEATURE_COLUMNS)} columnas x {len(feats):,} filas")
    print(f"Guardado en {out_path}")
    print("\nNaN por feature (periodo de calentamiento):")
    print(feats[FEATURE_COLUMNS].isna().sum().to_string())
    print("\nUltimas filas:")
    print(feats[FEATURE_COLUMNS].tail(3).to_string())


if __name__ == "__main__":
    main()
