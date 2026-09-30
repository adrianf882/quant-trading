"""Fase 7: backtest realista de las senales del modelo.

Genera predicciones OUT-OF-SAMPLE con walk-forward, las convierte en posiciones
(long/flat y long/short) y las pasa por el motor de backtest (ejecucion en la
apertura de t+1, con comision + slippage). Compara contra buy & hold y SMA 20/50
sobre el MISMO periodo.

Uso:
    py -m src.evaluate
"""
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para guardar PNG
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import run_backtest
from .baselines import buy_and_hold_signal, sma_cross_signal
from .config import get_paths, load_config
from .data import load_ohlcv, raw_data_path
from .features import FEATURE_COLUMNS, build_features
from .labels import triple_barrier
from .metrics import PERIODS_PER_YEAR, format_summary, summarize, trade_log
from .models import build_dataset
from .validation import walk_forward, walk_forward_predictions


def main() -> None:
    cfg = load_config()
    data_cfg, val_cfg, model_cfg, bt_cfg = (
        cfg["data"], cfg["validation"], cfg["model"], cfg["backtest"])
    symbol, timeframe = data_cfg["symbol"], data_cfg["timeframe"]
    ppy = PERIODS_PER_YEAR[timeframe]

    df = load_ohlcv(raw_data_path(symbol, timeframe))
    features = build_features(df)
    labels = triple_barrier(df, k=cfg["labels"]["k"], horizon=cfg["labels"]["horizon"],
                            vol_window=cfg["labels"]["vol_window"])
    data = build_dataset(features, labels, FEATURE_COLUMNS)

    # Solo trabajamos con dev; el holdout final sigue intocable (Fase 8).
    dev = data.iloc[: int(len(data) * (1 - val_cfg["holdout_pct"]))]
    X = dev[FEATURE_COLUMNS]
    y = dev["label"].to_numpy()
    w = dev["weight"].to_numpy()
    index, label_end = dev.index, dev["label_end"]

    splits = list(walk_forward(index, label_end, val_cfg["n_splits"],
                               val_cfg["embargo_pct"], min_train=5000))
    preds = walk_forward_predictions(model_cfg["params"], X, y, w, splits)
    eval_index = preds.dropna().index
    df_eval = df.loc[eval_index]
    print(f"Predicciones OOS: {len(eval_index):,} velas "
          f"({eval_index[0]} -> {eval_index[-1]})")

    signals = {
        "Buy & Hold": buy_and_hold_signal(df).reindex(eval_index),
        "SMA 20/50": sma_cross_signal(df, 20, 50).reindex(eval_index),
        "ML long/flat": (preds > 0).astype(float).reindex(eval_index),
        "ML long/short": preds.reindex(eval_index),
    }

    runs = {name: run_backtest(df_eval, sig, fee=bt_cfg["fee"],
                               slippage=bt_cfg["slippage"],
                               initial_capital=bt_cfg["initial_capital"])
            for name, sig in signals.items()}
    # Variante sin costos del modelo, para medir el impacto de las comisiones.
    runs["ML long/flat (sin costos)"] = run_backtest(
        df_eval, signals["ML long/flat"], fee=0.0, slippage=0.0,
        initial_capital=bt_cfg["initial_capital"])

    summaries = {name: summarize(bt, ppy) for name, bt in runs.items()}
    table = pd.DataFrame([format_summary(n, m) for n, m in summaries.items()])
    print(f"\nBacktest {symbol} {timeframe} | fee={bt_cfg['fee']} slippage={bt_cfg['slippage']}")
    print(f"Periodo OOS: {eval_index[0]} -> {eval_index[-1]}")
    print(table.to_string(index=False))

    # --- Guardado de resultados ---
    paths = get_paths()
    exp = paths["experiments"]
    exp.mkdir(parents=True, exist_ok=True)
    stamp = f"{symbol.replace('/', '_')}_{timeframe}"

    equity = pd.concat({name: bt["equity"] for name, bt in runs.items()}, axis=1)
    equity.to_csv(exp / f"equity_{stamp}.csv")

    trades = trade_log(runs["ML long/flat"]["ret"], runs["ML long/flat"]["pos"])
    trades.to_csv(exp / f"trades_{stamp}.csv", index=False)
    print(f"\nOperaciones ML long/flat: {len(trades)}")
    if len(trades):
        best = trades.loc[trades["retorno"].idxmax()]
        worst = trades.loc[trades["retorno"].idxmin()]
        print(f"  mejor: {best['retorno']:+.2%} ({best['entrada']} -> {best['salida']})")
        print(f"  peor : {worst['retorno']:+.2%} ({worst['entrada']} -> {worst['salida']})")

    ax = equity.div(equity.iloc[0]).plot(figsize=(11, 5), logy=True)
    ax.set_title(f"Curva de equity (OOS) - {symbol} {timeframe}")
    ax.set_ylabel("Capital (normalizado, escala log)")
    plt.tight_layout()
    fig_path = exp / f"equity_{stamp}.png"
    plt.savefig(fig_path, dpi=110)
    print(f"\nGuardado: {exp / f'equity_{stamp}.csv'}, {fig_path}, trades_{stamp}.csv")

    # --- Log de experimento (regla anti data-snooping: registrar cada corrida) ---
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fase": 7,
        "symbol": symbol, "timeframe": timeframe,
        "model_params": model_cfg["params"],
        "labels": cfg["labels"],
        "validation": val_cfg,
        "oos_period": [str(eval_index[0]), str(eval_index[-1])],
        "metrics": {name: {k: float(v) for k, v in m.items()} for name, m in summaries.items()},
    }
    with open(exp / f"phase7_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Log de experimento: {exp / f'phase7_{stamp}.json'}")


if __name__ == "__main__":
    main()
