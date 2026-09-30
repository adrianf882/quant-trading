"""Retorno esperado a 30 dias por estrategia (distribucion historica).

No es una prediccion: es la distribucion de retornos a 30 dias observada
historicamente (ventanas solapadas). Sirve para dimensionar la EXPECTATIVA y su
dispersion, no para prometer un numero.

Uso:
    py -m src.forecast
"""
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .baselines import sma_cross_signal
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .pairs import align
from .portfolio import build_signals, combined_backtest, sticky_bull

DAYS = 30


def rolling_return(r: pd.Series, window: int) -> pd.Series:
    """Retorno compuesto en ventanas de `window` barras (solapadas)."""
    log1p = np.log1p(r)
    cs = log1p.cumsum()
    return (np.exp(cs - cs.shift(window)) - 1).dropna()


def stats(name, r30):
    recent = r30[r30.index >= (r30.index[-1] - pd.Timedelta(days=730))]
    return {
        "estrategia": name,
        "media_30d": f"{r30.mean():+.2%}",
        "mediana_30d": f"{r30.median():+.2%}",
        "desvio": f"{r30.std():.1%}",
        "%positivos": f"{(r30 > 0).mean():.0%}",
        "p5": f"{r30.quantile(0.05):+.1%}",
        "p95": f"{r30.quantile(0.95):+.1%}",
        "peor": f"{r30.min():+.1%}",
        "mejor": f"{r30.max():+.1%}",
        "media_ult_2a": f"{recent.mean():+.2%}",
    }


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]

    # B-Kalman (futuros 4h): 30 dias = 180 velas.
    y, x = align("BTC/USDT", "ETH/USDT", "4h")
    bull = sticky_bull(y, 200, 0.02)
    sb, se, *_ = build_signals(y, x, 84, bull, 20, 50, hedge="kalman")
    bk = combined_backtest(y, x, sb, se, fee, slip, cap)["ret"]

    # Spot trend 1d: 30 dias = 30 velas.
    btc = load_ohlcv(raw_data_path("BTC/USDT", "1d"))
    trend = run_backtest(btc, sma_cross_signal(btc, 20, 50), fee, slip, cap)["ret"]
    bh = btc["close"].pct_change().fillna(0.0)

    rows = [
        stats("B-Kalman (futuros 4h)", rolling_return(bk, 180)),
        stats("SPOT trend 1d", rolling_return(trend, 30)),
        stats("BTC buy & hold 1d", rolling_return(bh, 30)),
    ]
    print(f"Retorno a {DAYS} dias - distribucion historica (ventanas solapadas)")
    print("(No es prediccion: es la expectativa bajo la historia.)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
