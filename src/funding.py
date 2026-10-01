"""Funding rate de perpetuos (Binance USDT-M) como variable de mercado.

- Descarga la historia de funding (gratis, sin key).
- Mide IC del funding (nivel, z-score, cambio) vs retornos a distintos horizontes.
- Prueba la estrategia CONTRARIAN: fade del lado hacinado cuando el z-score de
  funding es extremo (pos = -sign(z) si |z| > umbral).

Uso:
    py -m src.funding
"""
import ccxt
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import get_paths
from .data import load_ohlcv, raw_data_path
from .metrics import format_summary, summarize

PPY_8H = 365 * 3  # velas de 8h por ano


def fetch_funding(symbol="BTC/USDT:USDT", since="2019-09-01T00:00:00Z") -> pd.DataFrame:
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    since_ms = ex.parse8601(since)
    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
        if not batch:
            break
        rows += batch
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= since_ms:
            break
        since_ms = nxt
        if len(batch) < 1000:
            break
    df = pd.DataFrame([{"ts": r["timestamp"], "funding": r["fundingRate"]} for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("8h")
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def load_funding() -> pd.DataFrame:
    path = get_paths()["raw"] / "funding_btc.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch_funding().to_parquet(path)
    return pd.read_parquet(path)


def main() -> None:
    fund = load_funding()
    fund.index = pd.DatetimeIndex(fund.index)
    print(f"Funding BTC: {len(fund)} registros | {fund.index.min()} -> {fund.index.max()}")
    print(f"Funding medio: {fund['funding'].mean():.6f} por 8h -> ~{fund['funding'].mean()*PPY_8H:.1%} anual")

    # Precios BTC a 8h, alineados con los settlements de funding (00/08/16 UTC).
    btc = load_ohlcv(raw_data_path("BTC/USDT", "1h"))
    px = btc.resample("8h").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna()
    df = px.join(fund, how="inner")
    df["funding_z"] = (df["funding"] - df["funding"].rolling(90).mean()) / df["funding"].rolling(90).std()
    df["funding_chg"] = df["funding"].diff()
    df = df.dropna()

    print("\nIC del funding (Spearman) por horizonte:")
    for h, name in ((1, "8h"), (3, "24h"), (9, "72h"), (30, "10d")):
        fwd = np.log(df["close"]).diff(h).shift(-h)
        ics = {c: df[c].corr(fwd, method="spearman") for c in ("funding", "funding_z", "funding_chg")}
        print(f"  {name:4s}: " + "  ".join(f"{k}={v:+.3f}" for k, v in ics.items()))

    # --- Estrategia contrarian de funding extremo ---
    print("\nEstrategia contrarian (fade del hacinamiento), hold 3 velas (24h):")
    zero = pd.Series(0.0, index=df.index)
    rows = []
    for th in (1.5, 2.0, 2.5):
        raw = pd.Series(np.where(df["funding_z"] > th, -1.0,
                                 np.where(df["funding_z"] < -th, 1.0, 0.0)), index=df.index)
        held = raw.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)  # mantiene 3 velas
        bt = run_backtest(df, held, fee=0.0005, slippage=0.0005, initial_capital=10000.0)
        rows.append(format_summary(f"contrarian z>|{th}| (hold 24h)", summarize(bt, PPY_8H)))
    bh = run_backtest(df, pd.Series(1.0, index=df.index), fee=0.0005, slippage=0.0005,
                      initial_capital=10000.0)
    rows.append(format_summary("BTC B&H (8h)", summarize(bh, PPY_8H)))
    print(pd.DataFrame(rows).to_string(index=False))

    # --- Carry delta-neutral: long spot + short perp, se cobra el funding ---
    carry = df["funding"]                       # el short recibe si funding > 0
    eq = (1 + carry).cumprod()
    ann = eq.iloc[-1] ** (PPY_8H / len(df)) - 1
    sharpe = carry.mean() / carry.std() * np.sqrt(PPY_8H)
    dd = (eq / eq.cummax() - 1).min()
    print(f"\nCarry delta-neutral (long spot + short perp): "
          f"ret_anual~{ann:.1%} | Sharpe~{sharpe:.1f} | maxDD~{dd:.1%}")
    print(f"  (bruto, SIN costos de rebalanceo/basis; % funding negativo: {(carry < 0).mean():.0%})")


if __name__ == "__main__":
    main()
