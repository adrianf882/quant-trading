"""Fase 8: test final sobre el holdout intocable + auditoria honesta.

Configuracion CONGELADA (no se toca ningun parametro tras ver el resultado):
  - Timeframe: 4h
  - Regimen: sticky (close vs SMA200, banda +-2%)
  - Estrategia en alcista: SMA 20/50 long/flat sobre BTC
  - Estrategia en no-alcista: par BTC-ETH (lookback 84, entry 2.0, exit 0.5)
  - Costos: fee 0.001 + slippage 0.0005 por lado

Auditoria:
  1) Look-ahead: test de truncado (Chan) sobre las senales.
  2) Costos incluidos.
  3) Estabilidad dev vs holdout y por ano.
  4) Inventario de experimentos (para dimensionar data-snooping).

Uso:
    py -m src.audit
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .config import get_paths, load_config
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .pairs import align
from .portfolio import build_signals, combined_backtest, sticky_bull

FROZEN = {"timeframe": "4h", "window": 84, "band": 0.02, "fast": 20, "slow": 50}


def yearly(bt: pd.DataFrame) -> pd.Series:
    r = bt["ret"]
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def lookahead_audit(y, x, window, frac=0.9) -> bool:
    """Chan: las senales sobre datos truncados deben coincidir con las completas."""
    T = int(len(y) * frac)
    full_btc, full_eth, *_ = build_signals(y, x, window, sticky_bull(y))
    tr_btc, tr_eth, *_ = build_signals(y.iloc[:T], x.iloc[:T], window,
                                       sticky_bull(y.iloc[:T]))
    ok_btc = np.allclose(full_btc.iloc[:T], tr_btc, equal_nan=True)
    ok_eth = np.allclose(full_eth.iloc[:T], tr_eth, equal_nan=True)
    return bool(ok_btc and ok_eth)


def main() -> None:
    cfg = load_config()
    bt_cfg = cfg["backtest"]
    holdout = cfg["validation"]["holdout_pct"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]

    tf, window = FROZEN["timeframe"], FROZEN["window"]
    y, x = align("BTC/USDT", "ETH/USDT", tf)
    bull = sticky_bull(y, 200, FROZEN["band"])
    sig_btc, sig_eth, trend, pair_y, pair_x = build_signals(
        y, x, window, bull, FROZEN["fast"], FROZEN["slow"])

    cut = int(len(y) * (1 - holdout))
    ppy = PERIODS_PER_YEAR[tf]
    print("=" * 74)
    print("FASE 8 - TEST FINAL SOBRE HOLDOUT (una sola corrida)")
    print(f"Config congelada: {FROZEN} | costos fee={fee} slip={slip}")
    print(f"Dev: {y.index[0]} -> {y.index[cut-1]} | "
          f"Holdout: {y.index[cut]} -> {y.index[-1]}")
    print("=" * 74)

    def eval_period(name, sl):
        yp, xp = y.iloc[sl], x.iloc[sl]
        zero = pd.Series(0.0, index=yp.index)
        runs = {
            "Portfolio": combined_backtest(yp, xp, sig_btc.iloc[sl], sig_eth.iloc[sl], fee, slip, cap),
            "Tendencia": combined_backtest(yp, xp, trend.iloc[sl], zero, fee, slip, cap),
            "Par BTC-ETH": combined_backtest(yp, xp, pair_y.iloc[sl], pair_x.iloc[sl], fee, slip, cap),
            "BTC Buy & Hold": combined_backtest(yp, xp, pd.Series(1.0, index=yp.index), zero, fee, slip, cap),
        }
        print(f"\n--- {name} ---")
        print(pd.DataFrame([format_summary(n, summarize(bt, ppy)) for n, bt in runs.items()])
              .to_string(index=False))
        return runs

    dev_runs = eval_period("DEV (visto durante el desarrollo)", slice(0, cut))
    hold_runs = eval_period("HOLDOUT (intocable hasta ahora)", slice(cut, len(y)))

    pf_h = hold_runs["Portfolio"]
    print("\nRetorno del Portfolio por ano (dev + holdout):")
    combined = pd.concat([dev_runs["Portfolio"], hold_runs["Portfolio"]])
    print((yearly(combined) * 100).round(1).to_string())

    # --- Auditoria ---
    print("\n" + "=" * 74)
    print("AUDITORIA")
    print("=" * 74)
    ok = lookahead_audit(y, x, window)
    print(f"1) Look-ahead (truncado): {'OK - senales identicas' if ok else 'FALLA'}")

    exp = get_paths()["experiments"]
    files = sorted(exp.glob("*"))
    print(f"2) Costos: incluidos en cada pata (fee={fee} + slip={slip}) por lado")
    print(f"3) Inventario de experimentos en experiments/: {len(files)} archivos")
    fams = ["baselines", "features", "labels/triple-barrier", "LightGBM 3 clases",
            "purged k-fold + walk-forward", "meta-labeling", "timeframes 4h/1d",
            "dollar bars", "fracdiff", "regimen (bull/no-bull)", "short variants",
            "par BTC-ETH", "portfolio por regimen"]
    print(f"   familias de estrategias probadas: {len(fams)}")
    for f in fams:
        print(f"     - {f}")

    mh = summarize(pf_h, ppy)
    bh = summarize(hold_runs["BTC Buy & Hold"], ppy)
    verdict = ("SUPERA a buy&hold en Sharpe" if mh["sharpe"] > bh["sharpe"]
               else "NO supera a buy&hold en Sharpe")
    print(f"\nVeredicto holdout: Portfolio Sharpe={mh['sharpe']:.2f} vs "
          f"B&H Sharpe={bh['sharpe']:.2f} -> {verdict}")
    print("Caveat: config elegida entre 6 combinaciones (data-snooping); "
          "la muestra de holdout es una sola, sin repeticion.")

    exp.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "frozen_config": FROZEN,
        "costs": {"fee": fee, "slippage": slip},
        "lookahead_audit_ok": ok,
        "n_experiment_files": len(files),
        "families_tested": fams,
        "dev": {n: summarize(bt, ppy) for n, bt in dev_runs.items()},
        "holdout": {n: summarize(bt, ppy) for n, bt in hold_runs.items()},
        "yearly_portfolio": {int(k): float(v) for k, v in yearly(combined).items()},
    }
    with open(exp / "phase8_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nInforme JSON: {exp / 'phase8_report.json'}")


if __name__ == "__main__":
    main()
