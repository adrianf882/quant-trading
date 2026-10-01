"""Fase 9 (carry) - Paper trading del carry de funding (delta-neutral, 8h).

Mantiene: long spot BTC + short perp BTC. Cobra funding. Simula basis, costos,
rebalanceo mensual y capital partido. Wallet inicial: $1000.

Uso:
    py -m src.paper_carry
"""
import sys
from datetime import datetime

import ccxt
import pandas as pd

from .config import get_paths
from .paper import OHLCV

FROZEN = {"symbol": "BTC/USDT", "swap": "BTC/USDT:USDT", "timeframe": "8h",
          "initial_capital": 1000.0, "spot_fee": 0.001, "perp_fee": 0.0005,
          "margin_leverage": 3}
EXCHANGES = ["binanceusdm", "bybit"]


def detect_perp_exchange(swap: str):
    errors = []
    for ex_id in EXCHANGES:
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True,
                                       "options": {"defaultType": "swap"}})
            if ex.fetch_funding_rate_history(swap, limit=3):
                return ex
        except Exception as err:  # noqa: BLE001
            errors.append(f"{ex_id}: {type(err).__name__}")
    raise RuntimeError("Ningun exchange de perp respondio: " + ", ".join(errors))


def fetch_8h(ex, symbol, n=200):
    rows = ex.fetch_ohlcv(symbol, FROZEN["timeframe"], limit=n)
    df = pd.DataFrame(rows, columns=["ts", *OHLCV])[["ts", "close"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")["close"]


def fetch_funding(ex, swap, n=500):
    rows = ex.fetch_funding_rate_history(swap, limit=n)
    df = pd.DataFrame([{"ts": r["timestamp"], "funding": r["fundingRate"]} for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("8h")
    return df.drop_duplicates("ts").set_index("ts").sort_index()["funding"]


def main() -> None:
    log_path = get_paths()["experiments"] / "paper_log_carry.csv"
    cap = FROZEN["initial_capital"]

    ex = detect_perp_exchange(FROZEN["swap"])
    spot = fetch_8h(ex, FROZEN["symbol"])
    perp = fetch_8h(ex, FROZEN["swap"])
    fund = fetch_funding(ex, FROZEN["swap"])
    df = pd.DataFrame({"spot": spot, "perp": perp, "funding": fund}).dropna()
    bar = df.index[-1]
    print(f"Corrida: {datetime.now():%Y-%m-%d %H:%M} | exchange {ex.id} | vela {bar}")

    prev = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    if not prev.empty and str(prev["bar"].iloc[-1]) == str(bar):
        print(f"Sin vela nueva ({bar}); nada que registrar.")
        return

    notional = cap / (1 + 1 / FROZEN["margin_leverage"])
    if prev.empty:
        # Arranca AHORA: solo paga fees de entrada, sin backfill de historia.
        equity = cap - notional * (FROZEN["spot_fee"] + FROZEN["perp_fee"])
        q_spot = notional / df["spot"].iloc[-1]
        q_perp = notional / df["perp"].iloc[-1]
        start_idx = len(df)
    else:
        p0 = prev.iloc[-1]
        equity, q_spot, q_perp = p0["equity"], p0["q_spot"], p0["q_perp"]
        start_idx = df.index.get_loc(pd.Timestamp(p0["bar"])) + 1

    prev_s = df["spot"].iloc[start_idx - 1]
    prev_p = df["perp"].iloc[start_idx - 1]
    for i in range(start_idx, len(df)):
        s, p, f = df["spot"].iloc[i], df["perp"].iloc[i], df["funding"].iloc[i]
        equity += q_spot * (s - prev_s) - q_perp * (p - prev_p) + q_perp * p * f
        if df.index[i].month != df.index[i - 1].month:
            target = equity / (1 + 1 / FROZEN["margin_leverage"])
            q_spot, q_perp = target / s, target / p
        prev_s, prev_p = s, p

    ret = equity / cap - 1
    resumen = "QUIEBRA" if equity <= 0 else f"ret {ret:+.2%}"
    print(f"Funding ult: {fund.iloc[-1]:+.5%}/8h | equity: {equity:,.2f} ({resumen})")
    record = {"run_at": datetime.now().isoformat(timespec="seconds"), "bar": str(bar),
              "exchange": ex.id, "equity": equity, "q_spot": q_spot, "q_perp": q_perp,
              "funding": float(fund.iloc[-1])}
    pd.concat([prev, pd.DataFrame([record])]).to_csv(log_path, index=False)
    print(f"Log: {log_path} ({len(prev) + 1} registros)")


if __name__ == "__main__":
    sys.exit(main())
