"""Tests baratos pendientes (para no dejarlos sin probar):

1) SIZING por volatilidad (vol-target, emparentado con Kelly): la posicion long
   se escala por vol_objetivo/vol_realizada (cap 1, sin leverage). Chan cap. 5.
2) HEDGE por KALMAN: beta variable en el par BTC-ETH (vs OLS movil).
3) WALK-FORWARD con re-optimizacion movil de la MA (Chan l.656).
4) FUNDING: costo de carry en la exposicion neta (shorts de perpetuos).

Se evalua en todo el historico (dev+holdout, ya usado) y en 2023+.

Uso:
    py -m src.extra
"""
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .baselines import buy_and_hold_signal, sma_cross_signal
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .pairs import align, kalman_beta, rolling_hedge_ratio, pair_positions
from .portfolio import combined_backtest

FUNDING_PER_BAR_4H = 0.0001 / 2   # ~0.01% por 8h -> por vela de 4h
FUNDING_PER_BAR_1D = 0.0001 * 3   # ~0.01% por 8h -> por dia (3 ventanas)


def realized_vol(close, window, ppy):
    return np.log(close).diff().rolling(window).std() * np.sqrt(ppy)


def vol_target_signal(sig, close, window, ppy, cap=1.0):
    """Escala la senal long por (vol mediana historica / vol actual), con tope."""
    rv = realized_vol(close, window, ppy)
    scale = (rv.rolling(window * 3).median() / rv).clip(upper=cap).fillna(0.0)
    return sig * scale


def apply_funding(bt, pos_signed, rate_per_bar, cap):
    bt = bt.copy()
    bt["ret"] = bt["ret"] - pos_signed * rate_per_bar
    bt["equity"] = cap * (1 + bt["ret"]).cumprod()
    return bt


def wf_optimize(df, combos, ppy, train, rebal, fee, slip, cap):
    """Walk-forward: elige el mejor (fast,slow) en la ventana previa y lo aplica."""
    rets = {}
    for c in combos:
        rets[c] = run_backtest(df, sma_cross_signal(df, *c), fee, slip, cap)["ret"]
    rets = pd.DataFrame(rets)
    vals = rets.to_numpy()
    n = len(vals)
    chosen = np.full(n, np.nan)
    for t in range(train, n, rebal):
        window = vals[max(0, t - train):t]
        mean = np.nanmean(window, axis=0)
        std = np.nanstd(window, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe = np.where(std > 0, mean / std, -np.inf)
        best = int(np.argmax(sharpe))
        end = min(t + rebal, n)
        chosen[t:end] = vals[t:end, best]
    chosen = np.nan_to_num(chosen)
    chosen_ret = pd.Series(chosen, index=rets.index)
    equity = cap * (1 + chosen_ret).cumprod()
    return pd.DataFrame({"ret": chosen_ret, "equity": equity,
                         "pos": (chosen_ret != 0).astype(float)})


def show(name, bt, ppy):
    return format_summary(name, summarize(bt, ppy))


def section(title):
    print(f"\n{'='*76}\n{title}\n{'='*76}")


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]

    # ---------- 1) SIZING ----------
    section("1) SIZING por volatilidad (vol-target)")
    for tf, w in (("4h", 42), ("1d", 30)):
        df = load_ohlcv(raw_data_path("BTC/USDT", tf))
        ppy = PERIODS_PER_YEAR[tf]
        base = run_backtest(df, sma_cross_signal(df, 20, 50), fee, slip, cap)
        sig = vol_target_signal(sma_cross_signal(df, 20, 50), df["close"], w, ppy)
        sized = run_backtest(df, sig, fee, slip, cap)
        print(f"\n[{tf}] Trend 20/50")
        print(pd.DataFrame([show("fijo 0/1", base, ppy), show("vol-target", sized, ppy)])
              .to_string(index=False))

    # ---------- 2) KALMAN ----------
    section("2) HEDGE por Kalman en el par BTC-ETH")
    for tf, w in (("4h", 84), ("1d", 30)):
        y, x = align("BTC/USDT", "ETH/USDT", tf)
        ly, lx = np.log(y["close"]), np.log(x["close"])
        ppy = PERIODS_PER_YEAR[tf]
        for label, beta in (("OLS movil", rolling_hedge_ratio(ly, lx, w)),
                            ("Kalman", kalman_beta(ly, lx))):
            spread = ly - beta * lx
            z = (spread - spread.rolling(w).mean()) / spread.rolling(w).std()
            py, px = pair_positions(z, beta)
            bt = combined_backtest(y, x, py, px, fee, slip, cap)
            print(pd.DataFrame([show(f"{tf} {label}", bt, ppy)]).to_string(index=False))

    # ---------- 3) WALK-FORWARD MOVIL ----------
    section("3) Walk-forward con re-optimizacion movil de la MA")
    combos = [(f, s) for f in (5, 10, 20, 30, 50) for s in (50, 100, 150, 200) if f < s]
    for tf, train, rebal in (("4h", 2190, 180), ("1d", 365, 30)):
        df = load_ohlcv(raw_data_path("BTC/USDT", tf))
        ppy = PERIODS_PER_YEAR[tf]
        wf = wf_optimize(df, combos, ppy, train, rebal, fee, slip, cap)
        fixed = run_backtest(df, sma_cross_signal(df, 20, 50), fee, slip, cap)
        bh = run_backtest(df, buy_and_hold_signal(df), fee, slip, cap)
        print(f"\n[{tf}]")
        print(pd.DataFrame([show("WF movil", wf, ppy), show("fijo 20/50", fixed, ppy),
                            show("B&H", bh, ppy)]).to_string(index=False))

    # ---------- 4) FUNDING ----------
    section("4) Funding en la exposicion neta")
    for tf, rate, w in (("4h", FUNDING_PER_BAR_4H, 84), ("1d", FUNDING_PER_BAR_1D, 30)):
        y, x = align("BTC/USDT", "ETH/USDT", tf)
        ppy = PERIODS_PER_YEAR[tf]
        ly, lx = np.log(y["close"]), np.log(x["close"])
        beta = rolling_hedge_ratio(ly, lx, w)
        spread = ly - beta * lx
        z = (spread - spread.rolling(w).mean()) / spread.rolling(w).std()
        py, px = pair_positions(z, beta)
        pair = combined_backtest(y, x, py, px, fee, slip, cap)
        net = pair["pos_btc"] + pair["pos_eth"]
        pair_f = apply_funding(pair, net, rate, cap)
        # Tendencia larga: paga funding sobre la posicion long.
        trend = run_backtest(y, sma_cross_signal(y, 20, 50), fee, slip, cap)
        trend_f = apply_funding(trend, trend["pos"], rate, cap)
        print(f"\n[{tf}] rate/vela={rate:.6f}")
        print(pd.DataFrame([show("Par sin funding", pair, ppy), show("Par con funding", pair_f, ppy),
                            show("Trend sin funding", trend, ppy), show("Trend con funding", trend_f, ppy)])
              .to_string(index=False))


if __name__ == "__main__":
    main()
