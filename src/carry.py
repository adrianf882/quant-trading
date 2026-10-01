"""Carry de funding realista (delta-neutral: long spot + short perp).

Modela lo que el calculo bruto ignoraba:
  - BASIS real: usa precios spot Y perp (la pata corta no es exacta).
  - COSTOS: fees de entrada/salida en ambas patas + rebalanceo.
  - CAPITAL PARTIDO: la plata va a spot Y a margen del perp (sin apalancar el
    spot), asi que el retorno sobre el capital es menor que sobre el notional.

Uso:
    py -m src.carry
"""
import ccxt
import numpy as np
import pandas as pd

from .config import get_paths
from .data import load_ohlcv, raw_data_path

SPOT_FEE = 0.001    # taker spot
PERP_FEE = 0.0005   # taker perp
MARGIN_LEVERAGE = 3  # apalancamiento del margen del perp
PPY_8H = 365 * 3


def perp_8h() -> pd.DataFrame:
    path = get_paths()["raw"] / "perp_btc_8h.parquet"
    if not path.exists():
        ex = ccxt.binanceusdm({"enableRateLimit": True})
        since = ex.parse8601("2019-09-01T00:00:00Z")
        rows = []
        while True:
            batch = ex.fetch_ohlcv("BTC/USDT:USDT", "1h", since=since, limit=1000)
            if not batch:
                break
            rows += batch
            nxt = batch[-1][0] + 1
            if nxt <= since or len(batch) < 1000:
                break
            since = nxt
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").resample("8h").agg(
            {"open": "first", "close": "last"}).dropna()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    return df


def main() -> None:
    fund = pd.read_parquet(get_paths()["raw"] / "funding_btc.parquet")
    fund.index = pd.DatetimeIndex(fund.index)

    spot = load_ohlcv(raw_data_path("BTC/USDT", "1h")).resample("8h").agg(
        {"open": "first", "close": "last"}).dropna()
    perp = perp_8h()

    df = spot.join(perp, rsuffix="_perp").join(fund, how="inner").dropna()
    df["basis"] = df["close_perp"] / df["close"] - 1
    print(f"Carry BTC: {len(df)} velas 8h | {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Funding medio: {df['funding'].mean():.6f}/8h (~{df['funding'].mean()*PPY_8H:.1%}/ano) | "
          f"basis medio {df['basis'].mean():+.3%} | %funding<0: {(df['funding']<0).mean():.0%}")

    # --- Simulacion ---
    capital = 10000.0
    notional = capital / (1 + 1 / MARGIN_LEVERAGE)   # spot N + margen N/L = capital
    s0, p0 = df["close"].iloc[0], df["close_perp"].iloc[0]
    q_spot, q_perp = notional / s0, notional / p0     # long spot, short perp
    fees = notional * (SPOT_FEE + PERP_FEE)           # entrada
    equity = capital - fees
    hist, monthly_shorts = [capital], 0
    prev_s, prev_p = s0, p0
    for i in range(1, len(df)):
        s, p, f = df["close"].iloc[i], df["close_perp"].iloc[i], df["funding"].iloc[i]
        # P&L de precio: +spot -perp  (delta-neutral: cancela salvo por el basis)
        equity += q_spot * (s - prev_s) - q_perp * (p - prev_p)
        # funding: el short cobra si f > 0
        equity += q_perp * p * f
        # rebalanceo mensual para volver a delta-neutral (igualar notional)
        if df.index[i].month != df.index[i - 1].month:
            target = equity * (1 / (1 + 1 / MARGIN_LEVERAGE))
            q_spot, q_perp = target / s, target / p
            monthly_shorts += 1
        prev_s, prev_p = s, p
        hist.append(equity)

    eq = pd.Series(hist, index=df.index)
    ret = eq.pct_change().fillna(0.0)
    ann = (eq.iloc[-1] / capital) ** (PPY_8H / len(df)) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(PPY_8H)
    dd = (eq / eq.cummax() - 1).min()
    print("\nCarry REALISTA (capital en spot+margen, basis y costos, rebalanceo mensual):")
    print(f"  capital final: {eq.iloc[-1]:,.0f} (de {capital:,.0f})")
    print(f"  retorno anual: {ann:+.1%} | Sharpe: {sharpe:.2f} | maxDD: {dd:.1%} | "
          f"rebalanceos: {monthly_shorts}")
    print(f"  (notional usado: {notional:,.0f} = {notional/capital:.0%} del capital; "
          f"margen perp {MARGIN_LEVERAGE}x)")


if __name__ == "__main__":
    main()
