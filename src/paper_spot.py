"""Fase 9 (spot) - Paper trading de la tendencia long/flat en SPOT 1d.

Estrategia: SMA 20/50 long-only sobre BTC spot. Sin short, sin apalancamiento,
amigable con capital chico (paso spot ~$0,83). Log en paper_log_spot.csv.

Uso:
    py -m src.paper_spot
"""
import sys
from datetime import datetime

import pandas as pd

from .baselines import sma_cross_signal
from .config import get_paths, load_config
from .paper import detect_exchange, fetch

FROZEN = {"timeframe": "1d", "fast": 20, "slow": 50, "initial_capital": 100.0}


def main() -> None:
    cfg = load_config()
    log_path = get_paths()["experiments"] / "paper_log_spot.csv"
    fee = cfg["backtest"]["fee"]      # spot taker 0.1%
    slip = cfg["backtest"]["slippage"]

    ex, quote = detect_exchange(FROZEN["timeframe"])
    print(f"Exchange: {ex.id} (quote {quote})")
    btc = fetch(ex, f"BTC/{quote}", FROZEN["timeframe"])
    trend = sma_cross_signal(btc, FROZEN["fast"], FROZEN["slow"])

    bar = btc.index[-1]
    pos = float(trend.iloc[-1])
    price = float(btc["close"].iloc[-1])
    print(f"Corrida: {datetime.now():%Y-%m-%d %H:%M} | ultima vela diaria: {bar}")
    print(f"BTC={price:.1f} | posicion objetivo (0/1): {pos:.0f}")

    prev = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    if not prev.empty and str(prev["bar"].iloc[-1]) == str(bar):
        print(f"Sin vela nueva ({bar}); nada que registrar.")
        return
    equity = FROZEN["initial_capital"] if prev.empty else float(prev["equity"].iloc[-1])
    if not prev.empty:
        p0 = prev.iloc[-1]
        ret = price / p0["price_btc"] - 1
        turn = abs(pos - p0["pos_btc"])
        step = p0["pos_btc"] * ret - turn * (fee + slip)
        equity *= (1 + step)
        print(f"P&L del tramo: {step:+.3%} | equity simulada: {equity:,.2f}")

    record = {"run_at": datetime.now().isoformat(timespec="seconds"), "bar": str(bar),
              "exchange": ex.id, "pos_btc": pos, "price_btc": price, "equity": equity}
    pd.concat([prev, pd.DataFrame([record])]).to_csv(log_path, index=False)
    print(f"Log: {log_path} ({len(prev) + 1} registros)")


if __name__ == "__main__":
    sys.exit(main())
