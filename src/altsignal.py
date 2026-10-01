"""Estrategias basadas en atencion (Google Trends) y sentimiento (F&G).

Reglas simples long/flat (sin tunear umbrales finos), a baja frecuencia:
  - Atencion contrarian: long BTC si el z-score de busquedas "bitcoin" < 0.
  - Atencion extrema:    long solo si z < -1 (atencion muy baja).
  - F&G momentum:        long si F&G > 50.
  - Trend + atencion:    tendencia SMA 20/50 filtrada por z < 0.

CAVEAT: Google Trends se revisa retroactivamente (posible look-ahead); el
resultado es optimista. F&G no se revisa.

Uso:
    py -m src.altsignal
"""
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .baselines import sma_cross_signal
from .config import get_paths, load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, format_summary, summarize


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]
    ppy = PERIODS_PER_YEAR["1d"]

    btc = load_ohlcv(raw_data_path("BTC/USDT", "1d"))
    idx = btc.index

    gt = pd.read_parquet(get_paths()["raw"] / "gtrends.parquet")
    if gt.index.tz is None:
        gt.index = gt.index.tz_localize("UTC")
    gt_btc = gt["bitcoin"].reindex(idx, method="ffill").shift(7).astype(float)
    gt_z = (gt_btc - gt_btc.rolling(90).mean()) / gt_btc.rolling(90).std()

    fng = pd.read_parquet(get_paths()["raw"] / "fng.parquet")["fng"]
    fng_d = fng.reindex(idx, method="ffill").shift(1)

    trend = sma_cross_signal(btc, 20, 50)
    signals = {
        "Atencion contrarian (z<0)": (gt_z < 0).astype(float),
        "Atencion extrema (z<-1)": (gt_z < -1).astype(float),
        "F&G momentum (>50)": (fng_d > 50).astype(float),
        "F&G contrarian (<25)": (fng_d < 25).astype(float),
        "Trend + atencion (z<0)": trend * (gt_z < 0).astype(float),
    }
    # Variantes MENSUALES (el IC fuerte esta a 30 dias): decidir al inicio de mes.
    signals["Atencion contrarian MENSUAL"] = (
        (gt_z < 0).resample("MS").first().reindex(idx, method="ffill").astype(float).fillna(0.0))
    signals["Atencion extrema MENSUAL (z<-1)"] = (
        (gt_z < -1).resample("MS").first().reindex(idx, method="ffill").astype(float).fillna(0.0))
    signals["F&G momentum MENSUAL"] = (
        (fng_d > 50).resample("MS").first().reindex(idx, method="ffill").astype(float).fillna(0.0))
    runs = {name: run_backtest(btc, sig.fillna(0.0), fee, slip, cap)
            for name, sig in signals.items()}
    runs["Trend 20/50"] = run_backtest(btc, trend, fee, slip, cap)
    runs["BTC B&H"] = run_backtest(btc, pd.Series(1.0, index=idx), fee, slip, cap)

    print("Estrategias de atencion/sentimiento (BTC 1d, neto):")
    print(pd.DataFrame([format_summary(n, summarize(b, ppy)) for n, b in runs.items()])
          .to_string(index=False))


if __name__ == "__main__":
    main()
