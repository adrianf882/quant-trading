"""Portfolio B: regimen (tendencia en alcista / par en no-alcista) con hedge Kalman.

Compara B-OLS vs B-Kalman, tendencia sola, par-Kalman solo y B&H. Reporta total,
por ano y el split dev/holdout.

Uso:
    py -m src.portfolio_kalman
"""
import numpy as np
import pandas as pd

from .baselines import sma_cross_signal
from .config import load_config
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .pairs import align, kalman_beta, pair_positions
from .periods import yearly
from .portfolio import build_signals, combined_backtest, sticky_bull


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]
    holdout = cfg["validation"]["holdout_pct"]
    ppy = PERIODS_PER_YEAR["4h"]

    y, x = align("BTC/USDT", "ETH/USDT", "4h")
    cut = int(len(y) * (1 - holdout))
    bull = sticky_bull(y, 200, 0.02)
    zero = pd.Series(0.0, index=y.index)

    sb_ols, se_ols, trend, _, _ = build_signals(y, x, 84, bull, 20, 50, hedge="ols")
    sb_k, se_k, _, _, _ = build_signals(y, x, 84, bull, 20, 50, hedge="kalman")

    lbeta = kalman_beta(np.log(y["close"]), np.log(x["close"]))
    sp = np.log(y["close"]) - lbeta * np.log(x["close"])
    z = (sp - sp.rolling(84).mean()) / sp.rolling(84).std()
    py, px = pair_positions(z, lbeta)

    runs = {
        "B - OLS": combined_backtest(y, x, sb_ols, se_ols, fee, slip, cap),
        "B - Kalman": combined_backtest(y, x, sb_k, se_k, fee, slip, cap),
        "Trend 20/50": combined_backtest(y, x, trend, zero, fee, slip, cap),
        "Par Kalman": combined_backtest(y, x, py, px, fee, slip, cap),
        "BTC B&H": combined_backtest(y, x, pd.Series(1.0, index=y.index), zero, fee, slip, cap),
    }

    print("=" * 80)
    print("PORTFOLIO B (4h) - todo el historico")
    print("=" * 80)
    print(pd.DataFrame([format_summary(n, summarize(bt, ppy)) for n, bt in runs.items()])
          .to_string(index=False))

    print("\nRetorno por ano (%) - B-Kalman vs referencias:")
    print(pd.DataFrame({n: yearly(bt["ret"]) * 100 for n, bt in runs.items()}).round(1).to_string())

    print("\nB-Kalman: dev vs holdout")
    for name, sl in (("dev", slice(0, cut)), ("holdout", slice(cut, len(y)))):
        print(f"  {name}: {format_summary('B-Kalman', summarize(runs['B - Kalman'].iloc[sl], ppy))}")


if __name__ == "__main__":
    main()
