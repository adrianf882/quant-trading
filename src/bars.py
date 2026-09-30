"""Barras de informacion (Lopez de Prado, AFML cap. 2,2.3-2.5).

Las barras de tiempo son arbitrarias y de baja informacion (linea 995: samplear
en funcion de la actividad de trading da retornos mas cercanos a IID Normal).
Las **dollar bars** cierran una vela cada vez que se negocia un monto fijo de
dinero: se generan mas barras cuando hay actividad y menos cuando no.

Se construyen a partir de velas finas (5m); es una aproximacion estandar de las
dollar bars "verdaderas" (que usan cada trade), pero mantiene la propiedad de
samplear por actividad.

Uso:
    py -m src.bars    # construye dollar bars desde los 5m y las guarda
"""
import numpy as np
import pandas as pd

from .config import load_config
from .data import load_ohlcv, raw_data_path, save_ohlcv

SOURCE_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def build_dollar_bars(df: pd.DataFrame, target_minutes: int = 60,
                      window: int = 288, source_timeframe: str = "5m") -> pd.DataFrame:
    """Dollar bars: una vela cada `target_minutes` de actividad promedio.

    El umbral usa solo el pasado (`.shift(1)`), asi que no hay look-ahead.
    """
    src_min = SOURCE_MINUTES[source_timeframe]
    dollar_vol = (df["close"] * df["volume"]).to_numpy()
    dv = df["close"] * df["volume"]
    threshold = (dv.rolling(window).mean().shift(1) * (target_minutes / src_min)).to_numpy()

    o = h = l = c = None
    vol = 0.0
    accum = 0.0
    ts = df.index
    rows = []
    for i in range(len(df)):
        if accum == 0.0:
            o, h, l = df["open"].iat[i], df["high"].iat[i], df["low"].iat[i]
        accum += dollar_vol[i]
        h = max(h, df["high"].iat[i])
        l = min(l, df["low"].iat[i])
        c = df["close"].iat[i]
        vol += df["volume"].iat[i]
        t = threshold[i]
        if np.isfinite(t) and accum >= t:
            rows.append((ts[i], o, h, l, c, vol, accum))
            accum = 0.0
            vol = 0.0

    out = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low",
                                      "close", "volume", "dollar_volume"])
    return out.set_index("timestamp")


def main() -> None:
    cfg = load_config()
    symbol = cfg["data"]["symbol"]
    src = load_ohlcv(raw_data_path(symbol, "5m"))
    print(f"Origen 5m: {len(src):,} velas")

    bars = build_dollar_bars(src, target_minutes=60, window=288, source_timeframe="5m")
    out_path = raw_data_path(symbol, "dollar")
    save_ohlcv(bars.drop(columns=["dollar_volume"]), out_path)

    # Frecuencia promedio: cuantas barras por dia se generan (y dispersion).
    per_day = bars.groupby(bars.index.date).size()
    years = (bars.index[-1] - bars.index[0]).days / 365.25
    print(f"Dollar bars: {len(bars):,} ({bars.index.min()} -> {bars.index.max()})")
    print(f"Barras por dia: media={per_day.mean():.1f} min={per_day.min()} "
          f"max={per_day.max()} (dispersion = mas barras cuando hay actividad)")
    print(f"Guardadas en {out_path}")


if __name__ == "__main__":
    main()
