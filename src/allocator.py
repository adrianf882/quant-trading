"""Allocator por regimen "natural" del periodo (idea del usuario, robustecida).

Estadistico causal: AUTOCORRELACION lag-1 rolling de los retornos de BTC.
  - autocorr > 0  -> el mercado tiende a CONTINUAR  -> momentum/tendencia
  - autocorr <= 0 -> tiende a REVERTIR             -> mean-reversion (par BTC-ETH)

Peso a la tendencia w_t en {0,1} (o continuo); se asigna (1-w) al par. Todo
causal: el estadistico en `t` usa solo retornos hasta `t`; se ejecuta en `t+1`.

Se compara contra: tendencia sola, par solo, 50/50 fijo, y el Portfolio B (sticky).
Uso:
    py -m src.allocator
"""
import numpy as np
import pandas as pd

from .baselines import sma_cross_signal
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .extra import kalman_beta
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .pairs import align, pair_positions
from .periods import yearly
from .portfolio import build_signals, combined_backtest, sticky_bull

WINDOWS = [30, 60, 120]


def rolling_autocorr(close: pd.Series, window: int) -> pd.Series:
    """Autocorrelacion lag-1 rolling (vectorizada, causal)."""
    r = np.log(close).diff()
    rl = r.shift(1)
    mean_r = r.rolling(window).mean()
    mean_l = rl.rolling(window).mean()
    cov = (r * rl).rolling(window).mean() - mean_r * mean_l
    std = r.rolling(window).std() * rl.rolling(window).std()
    return cov / std


def allocator_run(y, x, window, fee, slip, cap, mode="sign", smooth_k=20.0):
    ppy = PERIODS_PER_YEAR["4h"]
    trend = sma_cross_signal(y, 20, 50)
    ly, lx = np.log(y["close"]), np.log(x["close"])
    beta = kalman_beta(ly, lx)
    spread = ly - beta * lx
    z = (spread - spread.rolling(84).mean()) / spread.rolling(84).std()
    pair_y, pair_x = pair_positions(z, beta)

    ac = rolling_autocorr(y["close"], window)
    if mode == "sign":
        w = (ac > 0).astype(float)
    elif mode == "smooth":
        w = (0.5 + ac * smooth_k).clip(0.0, 1.0)
    else:  # half
        w = pd.Series(0.5, index=y.index)
    w = w.fillna(0.0)

    pos_btc = w * trend + (1 - w) * pair_y.fillna(0.0)
    pos_eth = (1 - w) * pair_x.fillna(0.0)
    bt = combined_backtest(y, x, pos_btc, pos_eth, fee, slip, cap)
    bt.attrs["mean_w_trend"] = float(w.mean())
    return bt


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]
    ppy = PERIODS_PER_YEAR["4h"]

    y, x = align("BTC/USDT", "ETH/USDT", "4h")

    # Referencias.
    lsig = sma_cross_signal(y, 20, 50)
    ly, lx = np.log(y["close"]), np.log(x["close"])
    beta = kalman_beta(ly, lx)
    sp = ly - beta * lx
    z = (sp - sp.rolling(84).mean()) / sp.rolling(84).std()
    py, px = pair_positions(z, beta)
    zero = pd.Series(0.0, index=y.index)
    refs = {
        "Trend (100%)": combined_backtest(y, x, lsig, zero, fee, slip, cap),
        "Par Kalman (100%)": combined_backtest(y, x, py, px, fee, slip, cap),
        "50/50 fijo": combined_backtest(y, x, 0.5 * lsig, 0.5 * py.fillna(0), fee, slip, cap),
        "Portfolio B (sticky)": combined_backtest(
            y, x, *build_signals(y, x, 84, sticky_bull(y, 200, 0.02), 20, 50)[:2], fee, slip, cap),
    }

    print("=" * 80)
    print("ALLOCATOR por autocorrelacion rolling (4h)")
    print("=" * 80)
    rows = [format_summary(n, summarize(bt, ppy)) for n, bt in refs.items()]
    for window in WINDOWS:
        for mode in ("sign", "smooth"):
            bt = allocator_run(y, x, window, fee, slip, cap, mode=mode)
            name = f"Alloc {mode} w={window} (w_trend {bt.attrs['mean_w_trend']:.0%})"
            rows.append(format_summary(name, summarize(bt, ppy)))
    print(pd.DataFrame(rows).to_string(index=False))

    # Detalle por ano del mejor modo (sign, w=60).
    bt = allocator_run(y, x, 60, fee, slip, cap, mode="sign")
    print("\nAllocator sign w=60 vs referencias: retorno por ano (%)")
    print(pd.DataFrame({
        "alloc": yearly(bt["ret"]) * 100,
        "trend": yearly(refs["Trend (100%)"]["ret"]) * 100,
        "par": yearly(refs["Par Kalman (100%)"]["ret"]) * 100,
        "B&H-ish": yearly(refs["50/50 fijo"]["ret"]) * 100,
    }).round(1).to_string())


if __name__ == "__main__":
    main()
