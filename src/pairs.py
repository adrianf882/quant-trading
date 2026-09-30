"""Par trading / stat-arb BTC-ETH (Lopez de Prado cap. 6 / Chan).

Idea: si log(BTC) y log(ETH) estan cointegrados, el spread
    s_t = log(BTC_t) - beta * log(ETH_t)
es estacionario (revierte a la media). Se opera la reversión:
  - z alto  -> short spread (short BTC, long ETH)
  - z bajo  -> long spread  (long BTC, short ETH)
  - salir cerca de la media.

Todo causal: el hedge ratio `beta` y el z-score se estiman con ventana movil
(solo pasado); la senal se ejecuta en la apertura siguiente.

IMPORTANTE: es market-neutral (2 patas). El retorno no depende de que BTC suba,
sino de que el spread se cierre. Modela un short via perpetuos/margen SIN
funding (aproximacion).

Uso:
    py -m src.pairs
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .config import get_paths, load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .regimes import compounded, regime_labels, sharpe

# Ventana de estimacion por timeframe (aprox 1 semana en intradiario, 1 mes en diario).
LOOKBACK = {"1h": 168, "4h": 84, "1d": 30}
ENTRY_Z = 2.0   # a priori (Chan usa ~2 desvios)
EXIT_Z = 0.5    # a priori


def align(symbol_y: str, symbol_x: str, timeframe: str):
    y = load_ohlcv(raw_data_path(symbol_y, timeframe))
    x = load_ohlcv(raw_data_path(symbol_x, timeframe))
    common = y.index.intersection(x.index)
    return y.loc[common], x.loc[common]


def rolling_hedge_ratio(log_y: pd.Series, log_x: pd.Series, window: int) -> pd.Series:
    """Beta de OLS movil (regresion de log_y sobre log_x). Solo pasado."""
    cov = log_y.rolling(window).cov(log_x)
    var = log_x.rolling(window).var()
    return cov / var


def kalman_beta(log_y: pd.Series, log_x: pd.Series, delta: float = 1e-4,
                r: float = 1e-3) -> pd.Series:
    """Beta variable por filtro de Kalman 1D (estado = [alpha, beta]).

    Mas adaptativo que el OLS movil; el spread con beta de Kalman resulto mas
    rentable en las pruebas. Causal (solo pasado).
    """
    n = len(log_y)
    y = log_y.to_numpy()
    xv = log_x.to_numpy()
    theta = np.array([0.0, 1.0])
    P = np.eye(2)
    Q = np.eye(2) * delta / (1 - delta)
    out = np.full(n, np.nan)
    for t in range(n):
        if not (np.isfinite(y[t]) and np.isfinite(xv[t])):
            continue
        H = np.array([1.0, xv[t]])
        P = P + Q
        S = float(H @ P @ H + r)
        K = (P @ H) / S
        theta = theta + K * (y[t] - float(H @ theta))
        P = P - np.outer(K, H) @ P
        out[t] = theta[1]
    return pd.Series(out, index=log_y.index)


def pair_positions(z: pd.Series, beta: pd.Series, entry: float = ENTRY_Z,
                   exit: float = EXIT_Z):
    """Senal de spread {-1,0,1} -> posiciones por pata, beta-neutrales y normalizadas."""
    state = 0
    n = len(z)
    sy = np.zeros(n)
    sx = np.zeros(n)
    zz = z.to_numpy()
    bb = beta.to_numpy()
    for t in range(n):
        zv = zz[t]
        if np.isfinite(zv):
            if state == 0:
                if zv > entry:
                    state = -1      # spread caro -> short spread
                elif zv < -entry:
                    state = 1       # spread barato -> long spread
            elif state == 1 and zv >= -exit:
                state = 0
            elif state == -1 and zv <= exit:
                state = 0
        b = bb[t]
        g = (1 + abs(b)) if np.isfinite(b) else np.nan
        sy[t] = state / g
        sx[t] = -state * b / g
    return (pd.Series(sy, index=z.index, name="pos_y"),
            pd.Series(sx, index=z.index, name="pos_x"))


def run_pair_backtest(dfy: pd.DataFrame, dfx: pd.DataFrame, sig_y: pd.Series,
                      sig_x: pd.Series, fee: float, slippage: float,
                      initial_capital: float) -> pd.DataFrame:
    """Backtest de 2 patas: ejecucion en apertura t+1 y costos en AMBAS patas."""
    pos_y = sig_y.shift(1).fillna(0.0)
    pos_x = sig_x.shift(1).fillna(0.0)
    ret_y = dfy["open"].pct_change().shift(-1)
    ret_x = dfx["open"].pct_change().shift(-1)
    turn = pos_y.diff().abs().fillna(0.0) + pos_x.diff().abs().fillna(0.0)
    cost = turn * (fee + slippage)
    strat_ret = (pos_y * ret_y + pos_x * ret_x - cost).fillna(0.0)
    equity = initial_capital * (1 + strat_ret).cumprod()
    gross = pos_y.abs() + pos_x.abs()
    out = pd.DataFrame({"pos_y": pos_y, "pos_x": pos_x, "ret": strat_ret,
                        "equity": equity, "pos": gross})
    return out.iloc[:-1]


def adf_pvalue(series: pd.Series) -> float:
    s = series.dropna()
    return float(adfuller(s, maxlag=1, regression="c", autolag=None)[1])


def main() -> None:
    cfg = load_config()
    sym_y, sym_x = "BTC/USDT", "ETH/USDT"
    bt_cfg = cfg["backtest"]
    holdout = cfg["validation"]["holdout_pct"]
    rows = []

    for timeframe, window in LOOKBACK.items():
        y, x = align(sym_y, sym_x, timeframe)
        log_y, log_x = np.log(y["close"]), np.log(x["close"])
        beta = rolling_hedge_ratio(log_y, log_x, window)
        spread = log_y - beta * log_x
        z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

        corr = log_y.diff().corr(log_x.diff())

        # Solo dev (80%); el holdout final queda intocable.
        cut = int(len(y) * (1 - holdout))
        y_dev, x_dev = y.iloc[:cut], x.iloc[:cut]
        sy, sx = pair_positions(z.iloc[:cut], beta.iloc[:cut])
        bt = run_pair_backtest(y_dev, x_dev, sy, sx, bt_cfg["fee"],
                               bt_cfg["slippage"], bt_cfg["initial_capital"])

        ppy = PERIODS_PER_YEAR[timeframe]
        m = summarize(bt, ppy)
        # Benchmark: BTC buy & hold en el mismo periodo.
        bh = run_pair_backtest(y_dev, x_dev, pd.Series(1.0, index=y_dev.index),
                               pd.Series(0.0, index=x_dev.index), bt_cfg["fee"],
                               bt_cfg["slippage"], bt_cfg["initial_capital"])
        bh_m = summarize(bh, ppy)

        print(f"\n{'='*68}\nPAR BTC-ETH {timeframe} | lookback={window} entry={ENTRY_Z} exit={EXIT_Z}")
        print(f"Correlacion de retornos: {corr:.3f} | beta media={beta.mean():.2f} "
              f"(min={beta.min():.2f} max={beta.max():.2f})")
        print(f"ADF del spread (p<0.05 => estacionario): p={adf_pvalue(spread):.4f}")
        print("Estrategia par:")
        print(pd.DataFrame([format_summary("Par BTC-ETH", m)]).to_string(index=False))
        print("Benchmark BTC buy & hold (mismo periodo):")
        print(pd.DataFrame([format_summary("BTC B&H", bh_m)]).to_string(index=False))

        # Regimen (de BTC): el par deberia ser independiente de la direccion de BTC.
        reg = regime_labels(y_dev)
        common = bt.index
        bull = reg["bull"].reindex(common).fillna(False)
        r = bt["ret"]
        print("Por regimen de BTC:")
        print(f"  alcista:     ret={compounded(r[bull]):+.1%}  sharpe={sharpe(r[bull], ppy):.2f}")
        print(f"  no alcista:  ret={compounded(r[~bull]):+.1%}  sharpe={sharpe(r[~bull], ppy):.2f}")

        for name, mm in (("Par BTC-ETH", m), ("BTC B&H", bh_m)):
            rows.append({"timeframe": timeframe, **format_summary(name, mm)})

    exp = get_paths()["experiments"]
    pd.DataFrame(rows).to_csv(exp / "pairs_summary.csv", index=False)
    with open(exp / "pairs_summary.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {exp / 'pairs_summary.csv'}")


if __name__ == "__main__":
    main()
