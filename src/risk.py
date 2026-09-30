"""Gestión de capital / riesgo (Chan, cap. 5) sobre las estrategias.

Capas aplicadas:
  1) SIZING FRACCIONAL: invertir solo una fraccion del capital (ej. 50%).
  2) ENTRADA ESCALONADA: mover la posicion hacia el objetivo de a tramos
     (max `step` por vela), en vez de entrar todo de golpe.
  3) CORTACIRCUITOS DE DRAWDOWN: si la equity cae mas de `dd` desde el maximo,
     salir y esperar hasta recuperar por encima de `recover`.
  4) RIESGO POR VOLATILIDAD: escalar la posicion por vol_objetivo/vol_realizada
     (cap 1, sin leverage).

Se comparan capas por separado y todas juntas, sobre la linea SPOT 1d y la
linea B-Kalman (futuros 4h).

Uso:
    py -m src.risk
"""
import numpy as np
import pandas as pd

from .baselines import sma_cross_signal
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, summarize
from .pairs import align
from .portfolio import build_signals, sticky_bull
from .small_capital import (FILTERS, FEE, SLIP, FUNDING_8H, SPOT_FILTERS, floor_step)

BASE = {"size": 1.0, "step": 1.0, "dd": 0.0, "recover": 0.0, "vol": False}

# Overlay RECOMENDADO (aplicado al paper trading): sizing 50% + entrada escalonada.
SIZE_FRAC = 0.5
ENTRY_STEP = 1 / 3


def overlay_step(prev, target, size=SIZE_FRAC, step=ENTRY_STEP):
    """Posicion gestionada: objetivo escalado por `size`, acercandose <= `step` por vela."""
    goal = float(target) * size
    return float(prev + np.clip(goal - prev, -step, step))


CONFIGS = [
    ("base (sin overlay)", {}),
    ("+sizing 50%", {"size": 0.5}),
    ("+escalonada (1/3)", {"size": 0.5, "step": 1 / 3}),
    ("+cortacircuito dd 25%", {"size": 0.5, "step": 1 / 3, "dd": 0.25, "recover": 0.10}),
    ("+riesgo por vol (todas)", {"size": 0.5, "step": 1 / 3, "dd": 0.25, "recover": 0.10, "vol": True}),
]


def vol_scale(close: pd.Series, ppy: int, window: int) -> np.ndarray:
    rv = np.log(close).diff().rolling(window).std() * np.sqrt(ppy)
    median = rv.rolling(window * 3).median()
    return (median / rv).clip(0.0, 1.0).fillna(0.0).to_numpy()


def run(opens, sigs, capital, legs_filter, fee, slip, funding_8h, overlay, vols=None, index=None):
    L, n = len(opens), len(opens[0])
    equity, peak = capital, capital
    brake = False
    qpos, fpos = [0.0] * L, [0.0] * L
    hist = np.full(n, capital)
    for t in range(1, n):
        equity += sum(qpos[l] * (opens[l][t] - opens[l][t - 1]) for l in range(L))
        if funding_8h and t % 2 == 0:
            equity -= funding_8h * sum(abs(qpos[l]) * opens[l][t] for l in range(L))
        peak = max(peak, equity)
        dd = equity / peak - 1
        if overlay["dd"]:
            if dd <= -overlay["dd"]:
                brake = True
            if brake and dd >= -overlay["recover"]:
                brake = False
        turn = 0.0
        for l in range(L):
            tgt = sigs[l][t - 1] if np.isfinite(sigs[l][t - 1]) else 0.0
            if overlay["vol"] and vols is not None:
                tgt *= vols[l][t - 1]
            tgt *= overlay["size"]
            if brake:
                tgt = 0.0
            fpos[l] += np.clip(tgt - fpos[l], -overlay["step"], overlay["step"])
            step, minn = legs_filter[l]
            dq = floor_step(fpos[l] * equity / opens[l][t], step)
            if dq != 0 and abs(dq * opens[l][t]) < minn:
                dq = qpos[l]
            turn += abs(dq - qpos[l]) * opens[l][t]
            qpos[l] = dq
        equity -= turn * (fee + slip)
        hist[t] = equity
    ret = np.zeros(n)
    ret[1:] = hist[1:] / hist[:-1] - 1
    return pd.DataFrame({"ret": ret, "equity": hist, "pos": np.abs(np.sign(ret))}, index=index)


def summarize_rows(name, bt, ppy):
    m = summarize(bt, ppy)
    return {"variante": name, "ret_total": f"{m['total_return']:.0%}",
            "ret_anual": f"{m['ann_return']:.0%}", "sharpe": f"{m['sharpe']:.2f}",
            "max_dd": f"{m['max_drawdown']:.0%}", "ops": m["n_trades"]}


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    fee, slip = bt_cfg["fee"], bt_cfg["slippage"]

    # --- SPOT 1d (tendencia long/flat) ---
    btc = load_ohlcv(raw_data_path("BTC/USDT", "1d"))
    ppy1 = PERIODS_PER_YEAR["1d"]
    trend = sma_cross_signal(btc, 20, 50).to_numpy()
    spot_legs = [(SPOT_FILTERS["btc_step"], SPOT_FILTERS["btc_minn"])]
    spot_vol = [vol_scale(btc["close"], ppy1, 30)]
    print("=" * 80)
    print("SPOT trend 1d - overlay de riesgo (capital $100)")
    print("=" * 80)
    rows = [summarize_rows(name, run([btc["open"].to_numpy()], [trend], 100.0, spot_legs,
                                     0.001, 0.0005, 0.0, {**BASE, **ov}, spot_vol,
                                     index=btc.index), ppy1)
            for name, ov in CONFIGS]
    print(pd.DataFrame(rows).to_string(index=False))

    # --- B-Kalman futuros 4h ---
    y, x = align("BTC/USDT", "ETH/USDT", "4h")
    ppy4 = PERIODS_PER_YEAR["4h"]
    bull = sticky_bull(y, 200, 0.02)
    sb, se, *_ = build_signals(y, x, 84, bull, 20, 50, hedge="kalman")
    fut_legs = [(FILTERS["btc_step"], FILTERS["btc_minn"]), (FILTERS["eth_step"], FILTERS["eth_minn"])]
    fut_vol = [vol_scale(y["close"], ppy4, 42), vol_scale(x["close"], ppy4, 42)]
    print("\n" + "=" * 80)
    print("B-Kalman futuros 4h - overlay de riesgo (capital $1.000)")
    print("=" * 80)
    rows = [summarize_rows(name, run([y["open"].to_numpy(), x["open"].to_numpy()],
                                     [sb.to_numpy(), se.to_numpy()], 1000.0, fut_legs,
                                     FEE, SLIP, FUNDING_8H, {**BASE, **ov}, fut_vol,
                                     index=y.index), ppy4)
            for name, ov in CONFIGS]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
