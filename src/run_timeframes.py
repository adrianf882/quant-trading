"""Compara el pipeline completo en distintas frecuencias (4h, 1d).

Para cada timeframe: descarga (si falta), construye features/labels, y corre
buy & hold, SMA 20/50, el modelo 3-clases (OOS) y meta-labeling con primario de
tendencia. Reporta metricas comparables.

Nota de diseno: se mantiene `horizon` y `vol_window` CONSTANTES EN BARRAS (=24)
en todas las frecuencias, para no introducir tuning por timeframe. La
consecuencia es que el horizonte economico cambia (24h en 1h, 4 dias en 4h,
24 dias en 1d); es una comparacion honesta "misma maquinaria, distinta
granularidad", no una busqueda de parametros.

Uso:
    py -m src.run_timeframes
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .baselines import buy_and_hold_signal, sma_cross_signal
from .config import get_paths, load_config
from .data import download_ohlcv, load_ohlcv, raw_data_path, save_ohlcv
from .features import FEATURE_COLUMNS, build_features
from .labels import meta_labels, triple_barrier
from .meta import META_THRESHOLD, meta_signal, walk_forward_binary_prob
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .models import average_uniqueness, build_dataset
from .validation import walk_forward, walk_forward_predictions

TIMEFRAMES = ["4h", "1d"]


def ensure_data(symbol: str, timeframe: str, since: str) -> pd.DataFrame:
    path = raw_data_path(symbol, timeframe)
    if not path.exists():
        print(f"  descargando {symbol} {timeframe}...")
        save_ohlcv(download_ohlcv(symbol, timeframe, since), path)
    return load_ohlcv(path)


def build_runs(df: pd.DataFrame, cfg: dict, ppy: int,
               features: pd.DataFrame | None = None,
               short_variants: bool = False) -> dict:
    """Devuelve el backtest completo (DataFrame por estrategia) sobre la ventana OOS."""
    lab, val, model_cfg, bt_cfg = (
        cfg["labels"], cfg["validation"], cfg["model"], cfg["backtest"])
    k, horizon, vw = lab["k"], lab["horizon"], lab["vol_window"]
    n_splits, embargo, holdout = val["n_splits"], val["embargo_pct"], val["holdout_pct"]

    if features is None:
        features = build_features(df)
    feature_cols = list(features.columns)
    labels = triple_barrier(df, k=k, horizon=horizon, vol_window=vw)
    data = build_dataset(features, labels, feature_cols)
    dev = data.iloc[: int(len(data) * (1 - holdout))]
    X, y, w = dev[feature_cols], dev["label"].to_numpy(), dev["weight"].to_numpy()
    index, label_end = dev.index, dev["label_end"]

    # Modelo 3 clases (OOS).
    min_train = max(1000, int(0.25 * len(dev)))
    psplits = list(walk_forward(index, label_end, n_splits, embargo, min_train=min_train))
    preds = walk_forward_predictions(model_cfg["params"], X, y, w, psplits)

    # Meta-labeling con primario de tendencia. side_l: long-only; side_s: long/short.
    trend = features.reindex(index)["dist_sma200"]
    side_l = pd.Series(np.where(trend > 0, 1.0, 0.0), index=index)
    side_l[trend.isna()] = np.nan

    def meta_for(side: pd.Series):
        meta = meta_labels(df, side.reindex(df.index), k=k, horizon=horizon,
                           vol_window=vw).reindex(index)
        Xm = features.reindex(index)[feature_cols].copy()
        Xm["side"] = side
        mask = side.notna() & (side != 0) & meta["meta_label"].notna()
        Xm, ym = Xm[mask], meta.loc[mask, "meta_label"].astype(int).to_numpy()
        me = meta.loc[mask, "meta_end"]
        wm = average_uniqueness(index, meta["meta_end"])[mask.to_numpy()]
        splits = list(walk_forward(Xm.index, me, n_splits, embargo,
                                   min_train=max(800, int(0.25 * len(Xm)))))
        prob = walk_forward_binary_prob(model_cfg["params"], Xm, ym, wm, splits)
        return meta, prob

    meta_l, meta_prob_l = meta_for(side_l)
    side_s = side_l
    if short_variants:
        side_s = pd.Series(np.where(trend > 0, 1.0, -1.0), index=index)
        side_s[trend.isna()] = np.nan
        meta_s, meta_prob_s = meta_for(side_s)

    # Ventana OOS contigua = interseccion de los tramos con predicciones.
    starts, ends = [meta_prob_l.dropna().index[0]], [meta_prob_l.dropna().index[-1]]
    if short_variants:
        starts.append(meta_prob_s.dropna().index[0])
        ends.append(meta_prob_s.dropna().index[-1])
    start, end = max(starts), min(ends)
    eval_index = index[(index >= start) & (index <= end)]
    df_eval = df.loc[eval_index]

    signals = {
        "Buy & Hold": buy_and_hold_signal(df).reindex(eval_index),
        "SMA 20/50": sma_cross_signal(df, 20, 50).reindex(eval_index),
        "ML long/flat": (preds > 0).astype(float).reindex(eval_index),
        "Meta trend": meta_signal(side_l, meta_prob_l.reindex(index),
                                  meta_l["meta_end"].reindex(index),
                                  META_THRESHOLD).reindex(eval_index),
    }
    if short_variants:
        sma_fast = df["close"].rolling(20).mean()
        sma_slow = df["close"].rolling(50).mean()
        sma_ls = pd.Series(np.where(sma_fast > sma_slow, 1.0, -1.0), index=df.index)
        sma_ls[sma_slow.isna()] = 0.0
        signals["ML long/short"] = preds.reindex(eval_index)
        signals["SMA long/short"] = sma_ls.reindex(eval_index)
        signals["Meta L/S"] = meta_signal(side_s, meta_prob_s.reindex(index),
                                          meta_s["meta_end"].reindex(index),
                                          META_THRESHOLD).reindex(eval_index)

    runs = {name: run_backtest(df_eval, sig, fee=bt_cfg["fee"],
                               slippage=bt_cfg["slippage"],
                               initial_capital=bt_cfg["initial_capital"])
            for name, sig in signals.items()}
    runs["Meta trend (sin costos)"] = run_backtest(
        df_eval, signals["Meta trend"], fee=0.0, slippage=0.0,
        initial_capital=bt_cfg["initial_capital"])

    return runs


def run_all(df: pd.DataFrame, cfg: dict, ppy: int, features: pd.DataFrame | None = None) -> dict:
    """Versión resumida de build_runs: metricas por estrategia."""
    runs = build_runs(df, cfg, ppy, features)
    return {name: summarize(bt, ppy) for name, bt in runs.items()}


def main() -> None:
    cfg = load_config()
    data_cfg = cfg["data"]
    symbol, since = data_cfg["symbol"], data_cfg["since"]

    all_rows = []
    for tf in TIMEFRAMES:
        print(f"\n=== {symbol} {tf} ===")
        df = ensure_data(symbol, tf, since)
        print(f"  {len(df):,} velas: {df.index.min()} -> {df.index.max()}")
        summaries = run_all(df, cfg, PERIODS_PER_YEAR[tf])
        for name, m in summaries.items():
            row = {"timeframe": tf, **format_summary(name, m)}
            all_rows.append(row)
        table = pd.DataFrame([format_summary(n, m) for n, m in summaries.items()])
        print(table.to_string(index=False))

    full = pd.DataFrame(all_rows)
    exp = get_paths()["experiments"]
    exp.mkdir(parents=True, exist_ok=True)
    full.to_csv(exp / "timeframes_summary.csv", index=False)
    with open(exp / "timeframes_summary.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "threshold": META_THRESHOLD, "rows": all_rows},
                  f, indent=2, ensure_ascii=False)
    print("\n=== RESUMEN TODOS LOS TIMEFRAMES ===")
    print(full.to_string(index=False))
    print(f"\nGuardado en {exp / 'timeframes_summary.csv'}")


if __name__ == "__main__":
    main()
