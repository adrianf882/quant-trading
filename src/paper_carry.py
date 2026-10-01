"""Fase 9 (carry) - Paper trading del carry de funding (delta-neutral, 8h).

Mantiene: long spot BTC + short perp BTC. Cobra funding.

Robustez: los PRECIOS se toman de la cadena spot que ya funciona (Kraken/Binance)
y solo el FUNDING de un exchange de perpetuos (Binance/Bytbit/OKX). Se aproxima
precio perp ~ precio spot (el basis medio es -0,016%, despreciable). Wallet: $1000.

Uso:
    py -m src.paper_carry
"""
import sys
import time
from datetime import datetime

import ccxt
import pandas as pd

from .config import get_paths
from .paper import detect_exchange as detect_spot, fetch

FROZEN = {"symbol": "BTC/USDT", "swap": "BTC/USDT:USDT",
          "initial_capital": 1000.0, "spot_fee": 0.001, "perp_fee": 0.0005,
          "margin_leverage": 3}
EXCHANGES = ["binanceusdm", "bybit", "okx"]


def detect_funding_exchange(swap: str):
    errors = []
    for ex_id in EXCHANGES:
        for _ in range(3):
            try:
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True,
                                           "options": {"defaultType": "swap"}})
                if ex.fetch_funding_rate_history(swap, limit=3):
                    return ex
                break
            except Exception as err:  # noqa: BLE001
                errors.append(f"{ex_id}: {type(err).__name__}")
                time.sleep(2)
    raise RuntimeError("Ningun exchange de funding respondio: " + ", ".join(errors))


def to_8h(ohlcv: pd.DataFrame) -> pd.Series:
    return ohlcv["close"].resample("8h").last().dropna()


def fetch_funding(ex, swap, n=500) -> pd.Series:
    rows = ex.fetch_funding_rate_history(swap, limit=n)
    df = pd.DataFrame([{"ts": r["timestamp"], "funding": r["fundingRate"]} for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("8h")
    return df.drop_duplicates("ts").set_index("ts").sort_index()["funding"]


def main() -> None:
    log_path = get_paths()["experiments"] / "paper_log_carry.csv"
    cap = FROZEN["initial_capital"]

    ex_spot, quote = detect_spot("1h")
    spot = to_8h(fetch(ex_spot, f"BTC/{quote}", "1h"))
    ex_fund = detect_funding_exchange(FROZEN["swap"])
    fund = fetch_funding(ex_fund, FROZEN["swap"])
    print(f"Precios: {ex_spot.id} | funding: {ex_fund.id}")

    # Precio perp ~ precio spot (basis despreciable). Notional comun en 8h.
    df = pd.DataFrame({"spot": spot, "funding": fund}).dropna()
    bar = df.index[-1]
    print(f"Corrida: {datetime.now():%Y-%m-%d %H:%M} | vela {bar}")

    prev = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    if not prev.empty and str(prev["bar"].iloc[-1]) == str(bar):
        print(f"Sin vela nueva ({bar}); nada que registrar.")
        return

    notional = cap / (1 + 1 / FROZEN["margin_leverage"])
    if prev.empty:
        equity = cap - notional * (FROZEN["spot_fee"] + FROZEN["perp_fee"])
        q = notional / df["spot"].iloc[-1]
        start = len(df)
    else:
        p0 = prev.iloc[-1]
        equity, q = p0["equity"], p0["q"]
        start = df.index.get_loc(pd.Timestamp(p0["bar"])) + 1

    for i in range(start, len(df)):
        s, f = df["spot"].iloc[i], df["funding"].iloc[i]
        # delta-neutral con perp~spot: el P&L de precio se cancela; solo queda el funding.
        if df.index[i].month != df.index[i - 1].month:
            q = equity / (1 + 1 / FROZEN["margin_leverage"]) / s
        equity += q * s * f

    ret = equity / cap - 1
    estado = "QUIEBRA" if equity <= 0 else f"ret {ret:+.2%}"
    print(f"Funding ult: {fund.iloc[-1]:+.5%}/8h | equity: {equity:,.2f} ({estado})")
    invested = q * df["spot"].iloc[-1]                    # notional spot (long)
    margin = invested / FROZEN["margin_leverage"]         # margen en el perp
    liquidity = equity - invested - margin
    record = {"run_at": datetime.now().isoformat(timespec="seconds"), "bar": str(bar),
              "exchange": ex_fund.id, "equity": equity, "q": q, "funding": float(fund.iloc[-1]),
              "invested": invested, "liquidity": liquidity}
    pd.concat([prev, pd.DataFrame([record])]).to_csv(log_path, index=False)
    print(f"Log: {log_path} ({len(prev) + 1} registros)")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:  # noqa: BLE001
        print(f"Carry no disponible ahora ({type(err).__name__}: {err}). No bloquea el resto.")
        sys.exit(0)
