"""Evaluacion de datos alternativos (sentimiento) como variable predictiva.

1) IC de las features de Fear & Greed (nivel, cambio, z-score) vs el retorno
   siguiente.
2) Retorno por bucket de sentimiento (test contrarian: comprar con miedo).
3) Tendencia (SMA 20/50) vs tendencia + filtro de sentimiento.

Ojo look-ahead: el F&G se desplaza 1 dia antes de usarlo.

Uso:
    py -m src.altfeatures
"""
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .baselines import sma_cross_signal
from .config import get_paths, load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, format_summary, summarize


def load_fng() -> pd.Series:
    fng = pd.read_parquet(get_paths()["raw"] / "fng.parquet")
    return fng["fng"]


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]
    ppy = PERIODS_PER_YEAR["1d"]

    btc = load_ohlcv(raw_data_path("BTC/USDT", "1d"))
    fng = load_fng()

    df = pd.DataFrame(index=btc.index)
    df["close"] = btc["close"]
    # Sentimiento: usar el valor del dia ANTERIOR (publicado antes) -> shift 1.
    df["fng"] = fng.reindex(btc.index, method="ffill").shift(1)
    df = df.dropna()
    df["fng_chg"] = df["fng"].diff()
    df["fng_z"] = (df["fng"] - df["fng"].rolling(30).mean()) / df["fng"].rolling(30).std()
    df["fwd_ret"] = df["close"].pct_change().shift(-1)
    df = df.dropna()

    print("=" * 70)
    print("Fear & Greed como variable (BTC 1d)")
    print("=" * 70)
    print("IC (Spearman vs retorno siguiente):")
    for col in ("fng", "fng_chg", "fng_z"):
        print(f"  {col:8s}: {df[col].corr(df['fwd_ret'], method='spearman'):+.4f}")

    print("\nRetorno medio del dia siguiente por bucket de sentimiento:")
    bins = pd.cut(df["fng"], [-1, 24, 49, 74, 100],
                  labels=["Miedo extremo (0-24)", "Miedo (25-49)", "Neutral/Greed (50-74)", "Greed extremo (75-100)"])
    print(df.groupby(bins, observed=True)["fwd_ret"].agg(["mean", "count"]).to_string())

    # --- Google Trends semanal (shift 7d para evitar look-ahead) ---
    try:
        gt = pd.read_parquet(get_paths()["raw"] / "gtrends.parquet")
        if gt.index.tz is None:
            gt.index = gt.index.tz_localize("UTC")
        gt_d = gt.reindex(btc.index, method="ffill").shift(7)
        data2 = pd.DataFrame(index=btc.index)
        data2["fwd_ret"] = btc["close"].pct_change().shift(-1)
        for c in gt.columns:
            data2[c] = gt_d[c]
        data2 = data2.dropna()
        print("\nGoogle Trends (semanal) - IC vs retorno siguiente:")
        for c in gt.columns:
            print(f"  {c:14s}: {data2[c].corr(data2['fwd_ret'], method='spearman'):+.4f}")
    except FileNotFoundError:
        print("\n(Google Trends no disponible)")

    # --- Tendencia vs tendencia + filtro de sentimiento ---
    trend = sma_cross_signal(btc, 20, 50)
    filt = (df["fng"] < 80).reindex(btc.index).fillna(True)  # no estar long en greedy extremo
    trend_f = (trend * filt.astype(float))
    runs = {
        "Trend 20/50": run_backtest(btc, trend, fee, slip, cap),
        "Trend + filtro F&G<80": run_backtest(btc, trend_f, fee, slip, cap),
        "BTC B&H": run_backtest(btc, pd.Series(1.0, index=btc.index), fee, slip, cap),
    }
    print("\nTendencia vs tendencia + filtro de sentimiento:")
    print(pd.DataFrame([format_summary(n, summarize(b, ppy)) for n, b in runs.items()])
          .to_string(index=False))


if __name__ == "__main__":
    main()
