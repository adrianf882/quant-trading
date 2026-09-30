"""Portfolio por regimen: tendencia en alcista + par BTC-ETH en no-alcista.

Idea (Chan, "portfolio of strategies"): las dos estrategias son anti
correlacionadas por regimen. En vez de elegir una, se activa la que corresponde:
  - BTC alcista (close > SMA200)      -> SMA 20/50 long/flat sobre BTC
  - BTC no alcista (close <= SMA200)  -> par BTC-ETH (reversion del spread)

El regimen se conoce al cierre de `t` (causal), y se opera en `t+1`. El cambio
de estrategia paga costos de transicion en AMBAS patas.

Uso:
    py -m src.portfolio
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .baselines import sma_cross_signal
from .config import get_paths, load_config
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .pairs import LOOKBACK, align, kalman_beta, pair_positions, rolling_hedge_ratio
from .regimes import compounded, regime_labels


def combined_backtest(y: pd.DataFrame, x: pd.DataFrame, sig_btc: pd.Series,
                      sig_eth: pd.Series, fee: float, slippage: float,
                      initial_capital: float) -> pd.DataFrame:
    """Backtest de las dos patas (BTC y ETH) con costos y ejecucion en t+1."""
    pos_btc = sig_btc.shift(1).fillna(0.0)
    pos_eth = sig_eth.shift(1).fillna(0.0)
    ret_btc = y["open"].pct_change().shift(-1)
    ret_eth = x["open"].pct_change().shift(-1)
    turn = pos_btc.diff().abs().fillna(0.0) + pos_eth.diff().abs().fillna(0.0)
    cost = turn * (fee + slippage)
    strat_ret = (pos_btc * ret_btc + pos_eth * ret_eth - cost).fillna(0.0)
    equity = initial_capital * (1 + strat_ret).cumprod()
    gross = pos_btc.abs() + pos_eth.abs()
    out = pd.DataFrame({"pos_btc": pos_btc, "pos_eth": pos_eth, "ret": strat_ret,
                        "equity": equity, "pos": gross})
    return out.iloc[:-1]


def sticky_bull(y: pd.DataFrame, window: int = 200, band: float = 0.02) -> pd.Series:
    """Regimen alcista con hysteresis: entra si close > SMA*(1+band), sale si
    close < SMA*(1-band). Reduce el whipsaw de cruces de SMA200."""
    c = y["close"].to_numpy()
    s = y["close"].rolling(window).mean().to_numpy()
    state = False
    bull = np.zeros(len(c), dtype=bool)
    for i in range(len(c)):
        if not np.isfinite(s[i]):
            state = False
        elif not state and c[i] > s[i] * (1 + band):
            state = True
        elif state and c[i] < s[i] * (1 - band):
            state = False
        bull[i] = state
    return pd.Series(bull, index=y.index)


def build_signals(y, x, window, bull, fast=20, slow=50, hedge="ols"):
    log_y, log_x = np.log(y["close"]), np.log(x["close"])
    beta = rolling_hedge_ratio(log_y, log_x, window) if hedge == "ols" else kalman_beta(log_y, log_x)
    spread = log_y - beta * log_x
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    pair_y, pair_x = pair_positions(z, beta)
    trend = sma_cross_signal(y, fast, slow)
    bull = bull.fillna(False)

    sig_btc = pd.Series(np.where(bull, trend, pair_y.fillna(0.0)), index=y.index)
    sig_eth = pd.Series(np.where(bull, 0.0, pair_x.fillna(0.0)), index=y.index)
    return sig_btc, sig_eth, trend, pair_y, pair_x


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    holdout = cfg["validation"]["holdout_pct"]
    fee, slip = bt_cfg["fee"], bt_cfg["slippage"]
    cap = bt_cfg["initial_capital"]
    rows = []

    for tf, window in LOOKBACK.items():
        y, x = align("BTC/USDT", "ETH/USDT", tf)
        cut = int(len(y) * (1 - holdout))
        yv, xv = y.iloc[:cut], x.iloc[:cut]
        ppy = PERIODS_PER_YEAR[tf]
        zero = pd.Series(0.0, index=yv.index)

        regimes = {
            "plain (close>SMA200)": regime_labels(y)["bull"].fillna(False),
            "sticky (hysteresis 2%)": sticky_bull(y, 200, 0.02),
        }
        print(f"\n{'='*72}\nPORTFOLIO {tf} | lookback par={window}\n{'='*72}")

        for reg_name, bull in regimes.items():
            sig_btc, sig_eth, trend, pair_y, pair_x = build_signals(y, x, window, bull)
            bv = bull.iloc[:cut]
            flips = int((bv.astype(int).diff().abs() > 0).sum())
            runs = {
                "Tendencia (long)": combined_backtest(yv, xv, trend.iloc[:cut], zero, fee, slip, cap),
                "Par BTC-ETH": combined_backtest(yv, xv, pair_y.iloc[:cut], pair_x.iloc[:cut], fee, slip, cap),
                "Portfolio": combined_backtest(yv, xv, sig_btc.iloc[:cut], sig_eth.iloc[:cut], fee, slip, cap),
                "BTC Buy & Hold": combined_backtest(yv, xv, pd.Series(1.0, index=yv.index), zero, fee, slip, cap),
            }
            print(f"\n-- regimen {reg_name} | {bv.mean():.0%} alcista | {flips} cambios de regimen --")
            print(pd.DataFrame([format_summary(n, summarize(bt, ppy)) for n, bt in runs.items()])
                  .to_string(index=False))
            pf = runs["Portfolio"]; r = pf["ret"]
            print(f"   portfolio: alcista {compounded(r[bv]):+.0%} | no-alcista {compounded(r[~bv]):+.0%}"
                  f" | exposicion {float((pf['pos'] != 0).mean()):.0%}")
            for n, bt in runs.items():
                rows.append({"timeframe": tf, "regimen": reg_name,
                             **format_summary(n, summarize(bt, ppy))})

    exp = get_paths()["experiments"]
    pd.DataFrame(rows).to_csv(exp / "portfolio_summary.csv", index=False)
    with open(exp / "portfolio_summary.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"), "rows": rows},
                  f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {exp / 'portfolio_summary.csv'}")


if __name__ == "__main__":
    main()
