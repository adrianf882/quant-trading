"""Robustez de la estrategia de tendencia (Chan, cap. 3).

1) GRILLA DE PARAMETROS: corre SMA(fast/slow) long/flat sobre muchas
   combinaciones. Si el desempeno es estable -> robusto. Si solo anda en una
   combinacion -> sobreajuste (Chan l.694-696: "vary these parameters... see how
   the performance changes").

2) PORTFOLIO DE SISTEMAS (Chan cap. 1 y 6): promedia las senales de toda la
   grilla en una sola posicion -> diversifica el riesgo de elegir mal el
   parametro.

Uso:
    py -m src.robustness
"""
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .baselines import buy_and_hold_signal, sma_cross_signal
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, summarize

FASTS = [5, 10, 15, 20, 30, 50]
SLOWS = [50, 75, 100, 150, 200]


def main() -> None:
    cfg = load_config()
    data_cfg, bt_cfg = cfg["data"], cfg["backtest"]
    symbol = data_cfg["symbol"]
    holdout = cfg["validation"]["holdout_pct"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]

    for tf in ("4h", "1d", "1h"):
        df = load_ohlcv(raw_data_path(symbol, tf))
        dev = df.iloc[: int(len(df) * (1 - holdout))]
        ppy = PERIODS_PER_YEAR[tf]

        def run(sig):
            return run_backtest(dev, sig.reindex(dev.index), fee=fee,
                                slippage=slip, initial_capital=cap)

        bh = summarize(run(buy_and_hold_signal(df)), ppy)
        rows = []
        for fast in FASTS:
            for slow in SLOWS:
                if fast >= slow:
                    continue
                m = summarize(run(sma_cross_signal(df, fast, slow)), ppy)
                rows.append({"fast": fast, "slow": slow, "ret_anual": m["ann_return"],
                             "sharpe": m["sharpe"], "max_dd": m["max_drawdown"]})
        grid = pd.DataFrame(rows)

        # Portfolio: promedio de las senales de toda la grilla.
        sigs = [sma_cross_signal(df, f, s) for f in FASTS for s in SLOWS if f < s]
        port_sig = pd.concat(sigs, axis=1).mean(axis=1)
        mp = summarize(run(port_sig), ppy)

        print(f"\n{'='*70}\nROBUSTEZ TENDENCIA {tf} | dev | B&H Sharpe={bh['sharpe']:.2f} "
              f"ret={bh['ann_return']:+.1%}\n{'='*70}")
        print(f"Grilla ({len(grid)} combos): Sharpe media={grid['sharpe'].mean():.2f} "
              f"min={grid['sharpe'].min():.2f} max={grid['sharpe'].max():.2f} | "
              f"ret_anual media={grid['ret_anual'].mean():+.1%}")
        frac = (grid["sharpe"] > bh["sharpe"]).mean()
        print(f"Combos que baten a B&H en Sharpe: {frac:.0%}")
        print("\nPeores 3 y mejores 3 combos (Sharpe):")
        show = grid.sort_values("sharpe")
        print(pd.concat([show.head(3), show.tail(3)])[
            ["fast", "slow", "ret_anual", "sharpe", "max_dd"]].to_string(index=False))
        print(f"\nPortfolio de tendencias (senal promedio): ret={mp['ann_return']:+.1%} "
              f"Sharpe={mp['sharpe']:.2f} max_dd={mp['max_drawdown']:.1%} "
              f"ops={mp['n_trades']}")


if __name__ == "__main__":
    main()
