"""Meta-labeling (Lopez de Prado, AFML linea 2033) + baja rotacion.

Idea:
- Un modelo PRIMARIO decide el LADO de la apuesta (long/short). Puede tener
  alta recall y baja precision.
- Un modelo SECUNDARIO (meta-modelo) aprende a predecir si la apuesta primaria
  va a ser correcta (etiqueta binaria de `meta_labels`), y filtra los falsos
  positivos + decide si conviene apostar.

Para atacar el problema de Fase 7 (3290 operaciones que se comian el edge con
costos), ademas:
- Solo operamos si meta_prob > umbral.
- Mantenemos la posicion hasta la resolucion de la barrera (meta_end), en vez
  de re-decidir cada hora, lo que reduce drasticamente la rotacion.

Uso:
    py -m src.meta
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .baselines import buy_and_hold_signal, sma_cross_signal
from .config import get_paths, load_config
from .data import load_ohlcv, raw_data_path
from .features import FEATURE_COLUMNS, build_features
from .labels import meta_labels, triple_barrier
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .models import average_uniqueness, build_dataset, make_binary_model
from .validation import label_end_positions, walk_forward, walk_forward_predictions

META_THRESHOLD = 0.5  # a priori: solo apostamos si P(correcto) > 0.5


def walk_forward_binary_prob(params: dict, X: pd.DataFrame, y: np.ndarray,
                             w: np.ndarray, splits) -> pd.Series:
    """Probabilidad OUT-OF-SAMPLE P(meta-label=1) con walk-forward."""
    proba = pd.Series(np.nan, index=X.index, dtype=float)
    for train_pos, test_pos in splits:
        model = make_binary_model(params)
        model.fit(X.iloc[train_pos], y[train_pos], sample_weight=w[train_pos])
        proba.iloc[test_pos] = model.predict_proba(X.iloc[test_pos])[:, 1]
    return proba


def meta_signal(side: pd.Series, meta_prob: pd.Series, meta_end: pd.Series,
                threshold: float = META_THRESHOLD) -> pd.Series:
    """Construye la senal {-1,0,1}: apuesta si P>umbral y mantiene hasta meta_end.

    Devuelve la senal decidida al cierre de cada vela (el motor la ejecuta en t+1).
    Los tramos son NO solapados, asi que la rotacion es mucho menor.
    """
    n = len(side)
    end_pos = label_end_positions(side.index, meta_end)
    s = side.fillna(0.0).to_numpy()
    take = ((s != 0) & (meta_prob.fillna(0.0) > threshold).to_numpy()
            & meta_end.notna().to_numpy())
    signal = np.zeros(n)
    i = 0
    while i < n:
        if take[i]:
            j = end_pos[i]
            signal[i: j + 1] = s[i]  # mantengo hasta la resolucion
            i = j + 1                # siguiente apuesta: no solapa
        else:
            i += 1
    return pd.Series(signal, index=side.index, name="meta")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-labeling")
    parser.add_argument("--primary", choices=["model", "trend"], default="trend",
                        help="fuente del lado: 'model' (3 clases) o 'trend' (close>SMA200)")
    args = parser.parse_args()

    cfg = load_config()
    data_cfg, val_cfg, model_cfg, bt_cfg = (
        cfg["data"], cfg["validation"], cfg["model"], cfg["backtest"])
    lab = cfg["labels"]
    symbol, timeframe = data_cfg["symbol"], data_cfg["timeframe"]
    ppy = PERIODS_PER_YEAR[timeframe]

    df = load_ohlcv(raw_data_path(symbol, timeframe))
    features = build_features(df)
    labels = triple_barrier(df, k=lab["k"], horizon=lab["horizon"], vol_window=lab["vol_window"])
    data = build_dataset(features, labels, FEATURE_COLUMNS)

    dev = data.iloc[: int(len(data) * (1 - val_cfg["holdout_pct"]))]
    X, y, w = dev[FEATURE_COLUMNS], dev["label"].to_numpy(), dev["weight"].to_numpy()
    index, label_end = dev.index, dev["label_end"]

    # --- 1) PRIMARIO -> decide el lado ---
    if args.primary == "trend":
        trend = features.reindex(index)["dist_sma200"]
        side = pd.Series(np.where(trend > 0, 1.0, 0.0), index=index)
        side[trend.isna()] = np.nan
        print("Primario: tendencia long-only (close > SMA200)")
    else:
        psplits = list(walk_forward(index, label_end, val_cfg["n_splits"],
                                    val_cfg["embargo_pct"], min_train=5000))
        side = walk_forward_predictions(model_cfg["params"], X, y, w, psplits).reindex(index)
        print("Primario: clasificador 3 clases (OOS)")

    # --- 2) Etiquetas binarias de meta-labeling ---
    meta = meta_labels(df, side.reindex(df.index), k=lab["k"],
                       horizon=lab["horizon"], vol_window=lab["vol_window"]).reindex(index)

    # --- 3) Modelo SECUNDARIO (binario) -> P(apuesta correcta), OOS ---
    Xm = features.reindex(index)[FEATURE_COLUMNS].copy()
    Xm["side"] = side
    mask = side.notna() & (side != 0) & meta["meta_label"].notna()
    Xm, ym = Xm[mask], meta.loc[mask, "meta_label"].astype(int).to_numpy()
    meta_end = meta.loc[mask, "meta_end"]
    # Uniqueness sobre el indice COMPLETO y luego filtrada (evita posiciones -1).
    wm = average_uniqueness(index, meta["meta_end"])[mask.to_numpy()]

    msplits = list(walk_forward(Xm.index, meta_end, val_cfg["n_splits"],
                                val_cfg["embargo_pct"], min_train=3000))
    meta_prob = walk_forward_binary_prob(model_cfg["params"], Xm, ym, wm, msplits)
    print(f"Meta-muestras: {len(Xm):,} | tasa base P(correcta)={ym.mean():.3f}")

    # --- 4) Senal y backtest ---
    meta_prob_full = meta_prob.reindex(index)
    meta_end_full = meta["meta_end"].reindex(index)
    # OJO: hay que usar un rango CONTIGUO de velas. meta_prob tiene huecos
    # (filas donde no hay apuesta); pasar un indice con huecos a run_backtest
    # haria que pct_change calcule retornos entre velas no contiguas.
    meta_valid = meta_prob.dropna().index
    eval_index = index[(index >= meta_valid[0]) & (index <= meta_valid[-1])]
    df_eval = df.loc[eval_index]

    signals = {
        "Buy & Hold": buy_and_hold_signal(df).reindex(eval_index),
        "SMA 20/50": sma_cross_signal(df, 20, 50).reindex(eval_index),
        "Meta-labeling": meta_signal(side, meta_prob_full, meta_end_full,
                                     META_THRESHOLD).reindex(eval_index),
    }
    runs = {name: run_backtest(df_eval, sig, fee=bt_cfg["fee"],
                               slippage=bt_cfg["slippage"],
                               initial_capital=bt_cfg["initial_capital"])
            for name, sig in signals.items()}
    runs["Meta-labeling (sin costos)"] = run_backtest(
        df_eval, signals["Meta-labeling"], fee=0.0, slippage=0.0,
        initial_capital=bt_cfg["initial_capital"])

    summaries = {name: summarize(bt, ppy) for name, bt in runs.items()}
    table = pd.DataFrame([format_summary(n, m) for n, m in summaries.items()])
    print(f"\nMeta-labeling | primario={args.primary} | umbral={META_THRESHOLD} | "
          f"{symbol} {timeframe}")
    print(f"Periodo OOS: {eval_index[0]} -> {eval_index[-1]}")
    print(table.to_string(index=False))

    print("\nSensibilidad al umbral (solo informativo; el umbral se elige a priori):")
    for th in (0.5, 0.55, 0.6, 0.65):
        sig = meta_signal(side, meta_prob_full, meta_end_full, th).reindex(eval_index)
        bt = run_backtest(df_eval, sig, fee=bt_cfg["fee"], slippage=bt_cfg["slippage"],
                          initial_capital=bt_cfg["initial_capital"])
        m = summarize(bt, ppy)
        print(f"  umbral={th}: ops={m['n_trades']:4d} ret_anual={m['ann_return']:+.1%} "
              f"sharpe={m['sharpe']:+.2f} max_dd={m['max_drawdown']:.1%}")

    # --- Guardado ---
    exp = get_paths()["experiments"]; exp.mkdir(parents=True, exist_ok=True)
    stamp = f"{symbol.replace('/', '_')}_{timeframe}"
    pd.concat({n: bt["equity"] for n, bt in runs.items()}, axis=1).to_csv(
        exp / f"meta_equity_{stamp}.csv")
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fase": "7b_meta", "symbol": symbol, "timeframe": timeframe,
        "threshold": META_THRESHOLD,
        "model_params": model_cfg["params"], "labels": lab,
        "base_rate_meta": float(ym.mean()),
        "oos_period": [str(eval_index[0]), str(eval_index[-1])],
        "metrics": {n: {k: float(v) for k, v in m.items()} for n, m in summaries.items()},
    }
    with open(exp / f"phase7b_meta_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"\nLog: {exp / f'phase7b_meta_{stamp}.json'}")


if __name__ == "__main__":
    main()
