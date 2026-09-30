"""Analisis periodo por periodo ("ver menos para ver mejor").

El promedio de 7 anos mezcla fases incomparables (Chan, cap. 5: no-estacionariedad
y regime shifts; cap. 3 l.354: mirar con atencion los ultimos anos). Aca se reporta
el desempeno ANO POR ANO de cada estrategia, para ver en que fases gana cada una.

Uso:
    py -m src.periods
"""
import numpy as np
import pandas as pd

from .baselines import sma_cross_signal
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, max_drawdown
from .pairs import align
from .portfolio import build_signals, combined_backtest, sticky_bull


def yearly(r: pd.Series) -> pd.Series:
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def yearly_sharpe(r: pd.Series, ppy: int) -> pd.Series:
    def s(x):
        sd = x.std()
        return x.mean() / sd * np.sqrt(ppy) if sd and not np.isnan(sd) else np.nan
    return r.groupby(r.index.year).apply(s)


def build_4h(y, x, window, fee, slip, cap):
    bull = sticky_bull(y, 200, 0.02)
    sb, se, _, pair_y, pair_x = build_signals(y, x, window, bull, 20, 50)
    zero = pd.Series(0.0, index=y.index)
    return {
        "BTC B&H": combined_backtest(y, x, pd.Series(1.0, index=y.index), zero, fee, slip, cap),
        "Trend 20/50": combined_backtest(y, x, sma_cross_signal(y, 20, 50), zero, fee, slip, cap),
        "Trend 50/200": combined_backtest(y, x, sma_cross_signal(y, 50, 200), zero, fee, slip, cap),
        "Par BTC-ETH": combined_backtest(y, x, pair_y, pair_x, fee, slip, cap),
        "Portfolio 4h": combined_backtest(y, x, sb, se, fee, slip, cap),
    }


def report(tf: str, runs: dict, ppy: int) -> None:
    print(f"\n{'='*78}\nPERIODOS {tf}: retorno y Sharpe por ano\n{'='*78}")
    rets = pd.DataFrame({n: yearly(bt["ret"]) for n, bt in runs.items()}) * 100
    print("Retorno anual (%):")
    print(rets.round(1).to_string())

    print("\nSharpe por ano:")
    sh = pd.DataFrame({n: yearly_sharpe(bt["ret"], ppy) for n, bt in runs.items()})
    print(sh.round(2).to_string())

    print("\nResumen 2023+ (periodo reciente) vs total:")
    for n, bt in runs.items():
        r = bt["ret"]
        recent = r[r.index >= "2023-01-01"]
        print(f"  {n:14s} | total={(1+r).prod()-1:+7.1%} | 2023+={(1+recent).prod()-1:+7.1%} "
              f"| DD 2023+={max_drawdown(bt.loc[bt.index >= '2023-01-01','equity']):.1%}")


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]

    y, x = align("BTC/USDT", "ETH/USDT", "4h")
    runs = build_4h(y, x, 84, fee, slip, cap)
    report("4h", runs, PERIODS_PER_YEAR["4h"])

    # 1d por ano (tendencia y B&H).
    d = load_ohlcv(raw_data_path("BTC/USDT", "1d"))
    from .backtest import run_backtest
    zero = pd.Series(0.0, index=d.index)
    d1 = {
        "BTC B&H": run_backtest(d, pd.Series(1.0, index=d.index), fee, slip, cap),
        "Trend 20/50": run_backtest(d, sma_cross_signal(d, 20, 50), fee, slip, cap),
        "Trend 50/200": run_backtest(d, sma_cross_signal(d, 50, 200), fee, slip, cap),
    }
    report("1d", d1, PERIODS_PER_YEAR["1d"])


if __name__ == "__main__":
    main()
