"""IC de datos alternativos a distintos horizontes (1, 3, 5, 7 dias).

El test previo usaba solo el retorno del dia siguiente. Aca se mide el IC de
Fear & Greed y Google Trends contra retornos a 1, 3, 5 y 7 dias (horizonte).
Shift de 1 dia (F&G) y 7 dias (Trends semanal) para evitar look-ahead.

Uso:
    py -m src.althorizons
"""
import numpy as np
import pandas as pd

from .config import get_paths
from .data import load_ohlcv, raw_data_path


def build_df() -> pd.DataFrame:
    btc = load_ohlcv(raw_data_path("BTC/USDT", "1d"))
    out = pd.DataFrame(index=btc.index)
    out["close"] = btc["close"]

    fng = pd.read_parquet(get_paths()["raw"] / "fng.parquet")["fng"]
    out["fng"] = fng.reindex(btc.index, method="ffill").shift(1)
    out["fng_chg"] = out["fng"].diff()
    out["fng_z"] = (out["fng"] - out["fng"].rolling(30).mean()) / out["fng"].rolling(30).std()

    gt = pd.read_parquet(get_paths()["raw"] / "gtrends.parquet")
    if gt.index.tz is None:
        gt.index = gt.index.tz_localize("UTC")
    gt_d = gt.reindex(btc.index, method="ffill").shift(7)
    for c in gt.columns:
        if gt[c].nunique() > 1:
            out[f"gt_{c}"] = gt_d[c]
    return out


def main() -> None:
    df = build_df()
    feats = [c for c in df.columns if c != "close"]
    horizons = [1, 3, 5, 7, 14, 30]

    print("IC (Spearman) por horizonte - sentimiento/busquedas vs retorno a N dias:")
    table = {}
    for h in horizons:
        fwd = np.log(df["close"]).diff(h).shift(-h)
        table[f"h={h}d"] = {c: df[c].corr(fwd, method="spearman") for c in feats}
    print(pd.DataFrame(table).round(4).to_string())

    # Buckets de F&G con retorno a 3 y 7 dias.
    bins = pd.cut(df["fng"], [-1, 24, 49, 74, 100],
                  labels=["Miedo ext.", "Miedo", "Neutral/Greed", "Greed ext."])
    print("\nRetorno medio acumulado a 3 y 7 dias por bucket de F&G:")
    for h in (3, 7):
        fwd = df["close"].pct_change(h).shift(-h)
        g = pd.DataFrame({"fng_bucket": bins, "fwd": fwd}).dropna()
        print(f"\n  Horizonte {h}d:")
        print(g.groupby("fng_bucket", observed=True)["fwd"].agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    main()
