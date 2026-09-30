"""Fase 9 - Paper trading del Portfolio B-Kalman (dinero SIMULADO, datos reales).

Cada corrida:
  1) detecta un exchange accesible y descarga las ultimas velas 4h de BTC y ETH
     (descarta la vela en curso);
  2) reconstruye la configuracion CONGELADA (regimen sticky + tendencia + par-Kalman);
  3) calcula la posicion objetivo de la ultima vela cerrada;
  4) actualiza el P&L simulado y registra en CSV.

Se prueba una lista de exchanges porque Binance bloquea por geolocalizacion en
algunos datacenters (p. ej. runners de GitHub Actions -> HTTP 451).

No opera dinero real. Uso:
    py -m src.paper
"""
import sys
from datetime import datetime

import pandas as pd

from .config import get_paths, load_config
from .data import make_exchange
from .portfolio import build_signals, sticky_bull

FROZEN = {"timeframe": "4h", "window": 84, "band": 0.02, "fast": 20, "slow": 50,
          "hedge": "kalman", "initial_capital": 10000.0}
OHLCV = ["open", "high", "low", "close", "volume"]
CANDIDATES = [("binance", "USDT"), ("kraken", "USD"), ("coinbase", "USD"), ("okx", "USDT")]


def detect_exchange(timeframe: str):
    """Devuelve el primer exchange que responda, con su quote de confianza."""
    errors = []
    for ex_id, quote in CANDIDATES:
        try:
            ex = make_exchange(ex_id)
            if ex.fetch_ohlcv(f"BTC/{quote}", timeframe, limit=5):
                return ex, quote
        except Exception as err:  # noqa: BLE001  (probamos el siguiente)
            errors.append(f"{ex_id}: {type(err).__name__}")
    raise RuntimeError("Ningun exchange respondio: " + ", ".join(errors))


def fetch(ex, symbol: str, timeframe: str, n: int = 1000) -> pd.DataFrame:
    rows = ex.fetch_ohlcv(symbol, timeframe, limit=n)
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").iloc[:-1]  # descarta la vela en curso


def main() -> None:
    cfg = load_config()
    log_path = get_paths()["experiments"] / "paper_log.csv"
    fee = cfg["backtest"]["fee"]
    slip = cfg["backtest"]["slippage"]

    ex, quote = detect_exchange(FROZEN["timeframe"])
    print(f"Exchange: {ex.id} (quote {quote})")
    btc = fetch(ex, f"BTC/{quote}", FROZEN["timeframe"])
    eth = fetch(ex, f"ETH/{quote}", FROZEN["timeframe"])
    common = btc.index.intersection(eth.index)
    y, x = btc.loc[common], eth.loc[common]

    bull = sticky_bull(y, 200, FROZEN["band"])
    sb, se, _, _, _ = build_signals(y, x, FROZEN["window"], bull,
                                    FROZEN["fast"], FROZEN["slow"],
                                    hedge=FROZEN["hedge"])
    bar = y.index[-1]
    pos_btc, pos_eth = float(sb.iloc[-1]), float(se.iloc[-1])
    regime = "alcista" if bool(bull.iloc[-1]) else "no alcista"
    price_btc, price_eth = float(y["close"].iloc[-1]), float(x["close"].iloc[-1])

    print(f"Corrida: {datetime.now():%Y-%m-%d %H:%M} | ultima vela cerrada: {bar}")
    print(f"Regimen: {regime} | BTC={price_btc:.1f} ETH={price_eth:.1f}")
    print(f"Posicion objetivo -> BTC: {pos_btc:+.3f} | ETH: {pos_eth:+.3f}")

    prev = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    equity = FROZEN["initial_capital"] if prev.empty else float(prev["equity"].iloc[-1])
    if not prev.empty and str(prev["bar"].iloc[-1]) != str(bar):
        p0 = prev.iloc[-1]
        ret_btc = price_btc / p0["price_btc"] - 1
        ret_eth = price_eth / p0["price_eth"] - 1
        turn = abs(pos_btc - p0["pos_btc"]) + abs(pos_eth - p0["pos_eth"])
        step = p0["pos_btc"] * ret_btc + p0["pos_eth"] * ret_eth - turn * (fee + slip)
        equity *= (1 + step)
        print(f"P&L del tramo: {step:+.3%} | equity simulada: {equity:,.2f}")

    record = {"run_at": datetime.now().isoformat(timespec="seconds"), "bar": str(bar),
              "exchange": ex.id, "regime": regime, "pos_btc": pos_btc, "pos_eth": pos_eth,
              "price_btc": price_btc, "price_eth": price_eth, "equity": equity}
    pd.concat([prev, pd.DataFrame([record])]).to_csv(log_path, index=False)
    print(f"Log: {log_path} ({len(prev) + 1} registros)")


if __name__ == "__main__":
    sys.exit(main())
