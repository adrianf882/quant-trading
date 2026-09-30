"""Simulacion realista con capital chico (fricciones de exchange).

Modela lo que nuestro backtest fraccional NO captura:
  - tamano de lote (step size) y cantidad minima por orden
  - notional MINIMO por orden (BTC $50, ETH $20 en Binance USDT-M)
  - comision taker de futuros (0,05%) + slippage
  - funding cada 8h sobre el notional abierto

Compara la estrategia B-Kalman y la tendencia con 100 USD vs 10.000 USD, y
contra una version "ideal" fraccional (sin fricciones).

Uso:
    py -m src.small_capital
"""
import numpy as np
import pandas as pd

from .baselines import sma_cross_signal
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, summarize
from .pairs import align
from .portfolio import build_signals, combined_backtest, sticky_bull

FILTERS = {"btc_step": 0.001, "eth_step": 0.001, "btc_minn": 50.0, "eth_minn": 20.0}
SPOT_FILTERS = {"btc_step": 1e-5, "eth_step": 1e-4, "btc_minn": 5.0, "eth_minn": 5.0}
FEE = 0.0005      # taker Binance USDT-M
SLIP = 0.0005
FUNDING_8H = 0.0001


def floor_step(q, step):
    return np.sign(q) * np.floor(abs(q) / step + 1e-9) * step


def simulate(y, x, sig_btc, sig_eth, capital, filters=FILTERS, fee=FEE, slip=SLIP,
             funding_8h=FUNDING_8H):
    ob, oe = y["open"].to_numpy(), x["open"].to_numpy()
    sb, se = sig_btc.fillna(0).to_numpy(), sig_eth.fillna(0).to_numpy()
    n = len(ob)
    qb = qe = 0.0
    equity = capital
    hist = np.full(n, capital, dtype=float)
    trades = blocked_btc = blocked_eth = 0
    btc_flat_target = bars_target = btc_notional_sum = 0.0
    for t in range(1, n):
        equity += qb * (ob[t] - ob[t - 1]) + qe * (oe[t] - oe[t - 1])
        if t % 2 == 0:  # funding cada 8h (0 en spot)
            equity -= funding_8h * (abs(qb) * ob[t] + abs(qe) * oe[t])
        tgt_b, tgt_e = sb[t - 1] * equity, se[t - 1] * equity
        dqb = floor_step(tgt_b / ob[t], filters["btc_step"])
        dqe = floor_step(tgt_e / oe[t], filters["eth_step"])
        # pata BTC bloqueada: queriamos estar en BTC (|tgt|>=min) pero no alcanza el paso
        if abs(tgt_b) >= filters["btc_minn"] and dqb == 0:
            blocked_btc += 1
        if abs(tgt_e) >= filters["eth_minn"] and dqe == 0:
            blocked_eth += 1
        # diagnostico: queriamos BTC (target>min) pero quedamos planos
        if abs(tgt_b) > filters["btc_minn"]:
            bars_target += 1
            if abs(dqb) < 1e-12:
                btc_flat_target += 1
        btc_notional_sum += abs(qb) * ob[t]
        if dqb != 0 and abs(dqb * ob[t]) < filters["btc_minn"]:
            dqb = qb
        if dqe != 0 and abs(dqe * oe[t]) < filters["eth_minn"]:
            dqe = qe
        turn = abs(dqb - qb) * ob[t] + abs(dqe - qe) * oe[t]
        if turn > 0:
            trades += 1
        equity -= turn * (fee + slip)
        qb, qe = dqb, dqe
        hist[t] = equity
    ret = np.zeros(n)
    ret[1:] = hist[1:] / hist[:-1] - 1
    bt = pd.DataFrame({"ret": ret, "equity": hist, "pos": np.abs(np.sign(ret))}, index=y.index)
    bt.attrs.update(trades=trades, blocked_btc=blocked_btc, blocked_eth=blocked_eth,
                    avg_btc_notional=btc_notional_sum / n,
                    pct_btc_flat_when_target=btc_flat_target / max(bars_target, 1))
    return bt


def main() -> None:
    ppy = PERIODS_PER_YEAR["4h"]
    year = 365 * 6  # velas 4h en un ano
    y, x = align("BTC/USDT", "ETH/USDT", "4h")
    bull = sticky_bull(y, 200, 0.02)
    sb, se, trend, _, _ = build_signals(y, x, 84, bull, 20, 50, hedge="kalman")
    zero = pd.Series(0.0, index=y.index)

    def row(name, bt, extra=None, tf_ppy=ppy):
        m = summarize(bt, tf_ppy)
        m1 = summarize(bt.iloc[:year], tf_ppy)
        r = {"escenario": name, "ops": bt.attrs.get("trades", summarize(bt, tf_ppy)["n_trades"]),
             "ret_total": f"{m['total_return']:.0%}", "sharpe": f"{m['sharpe']:.2f}",
             "max_dd": f"{m['max_drawdown']:.0%}",
             "1er_anio": f"{m1['total_return']:+.1%}", "sharpe_1y": f"{m1['sharpe']:.2f}"}
        if extra:
            r.update(extra)
        return r

    rows = []
    for cap in (100.0, 1000.0, 10000.0):
        bt = simulate(y, x, sb, se, cap)
        rows.append(row(f"B-Kalman realista ${cap:.0f}", bt,
                        {"bloq_btc": bt.attrs["blocked_btc"], "bloq_eth": bt.attrs["blocked_eth"]}))
    for cap in (100.0, 10000.0):
        bt = combined_backtest(y, x, sb, se, FEE, SLIP, cap)
        rows.append(row(f"B-Kalman ideal (fraccional) ${cap:.0f}", bt))
    for cap in (100.0, 10000.0):
        bt = simulate(y, x, trend, zero, cap)
        rows.append(row(f"Trend 1-pata realista ${cap:.0f}", bt))

    # --- SPOT long-only: solo la pata de tendencia (sin short) ---
    for tf in ("1d", "4h"):
        btc = load_ohlcv(raw_data_path("BTC/USDT", tf))
        tr = sma_cross_signal(btc, 20, 50)
        zb = pd.Series(0.0, index=btc.index)
        for cap in (100.0, 10000.0):
            bt = simulate(btc, btc, tr, zb, cap, filters=SPOT_FILTERS,
                          fee=0.001, slip=0.0005, funding_8h=0.0)
            rows.append(row(f"SPOT trend {tf} ${cap:.0f}", bt,
                            tf_ppy=PERIODS_PER_YEAR[tf]))

    print("Fricciones: step BTC=0.001 ($~83) min=$50 | ETH=0.001 ($~2.7) min=$20 | "
          f"fee={FEE} slip={SLIP} funding 8h={FUNDING_8H}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
