"""Revision a la luz de Jansen (ML for Algorithmic Trading, 2e).

Jansen enfatiza: (a) evaluacion de factores por IC (Information Coefficient),
(b) ENSEMBLES de modelos en vez de uno solo, (c) CV temporal.

Aca:
  1) IC de cada feature contra el retorno de la vela siguiente (Spearman).
  2) Ensemble (RandomForest + GradientBoosting + LightGBM) con walk-forward OOS,
     vs el LightGBM solo. Se backtestea long/short y long/flat.
Ver si algo cambia respecto de lo ya visto (probablemente no).

Uso:
    py -m src.jansen
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .backtest import run_backtest
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .features import FEATURE_COLUMNS, build_features
from .labels import triple_barrier
from .metrics import PERIODS_PER_YEAR, format_summary, summarize
from .models import average_uniqueness, build_dataset
from .validation import walk_forward


def oos_proba(factory, X, y, w, splits, classes):
    proba = pd.DataFrame(np.nan, index=X.index, columns=classes)
    for tr, te in splits:
        m = factory()
        m.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        proba.iloc[te] = m.predict_proba(X.iloc[te])
    return proba


def main() -> None:
    cfg = load_config()
    model_params = cfg["model"]["params"]
    lab = cfg["labels"]
    bt_cfg = cfg["backtest"]
    val = cfg["validation"]
    fee, slip, cap = bt_cfg["fee"], bt_cfg["slippage"], bt_cfg["initial_capital"]

    for tf in ("1d", "4h"):
        df = load_ohlcv(raw_data_path("BTC/USDT", tf))
        feats = build_features(df)
        labels = triple_barrier(df, k=lab["k"], horizon=lab["horizon"], vol_window=lab["vol_window"])
        data = build_dataset(feats, labels, FEATURE_COLUMNS)
        dev = data.iloc[: int(len(data) * (1 - val["holdout_pct"]))]
        X = dev[FEATURE_COLUMNS]
        y = dev["label"]
        w = dev["weight"].to_numpy()
        index, label_end = dev.index, dev["label_end"]
        ppy = PERIODS_PER_YEAR[tf]

        # --- 1) IC de features vs retorno siguiente ---
        fwd = np.log(df["close"]).diff().shift(-1)
        ic = feats[FEATURE_COLUMNS].corrwith(fwd, method="spearman").sort_values()
        print(f"\n{'='*72}\nJANSEN {tf}: IC de features (Spearman vs retorno siguiente)\n{'='*72}")
        print(ic.round(4).to_string())

        # --- 2) Ensemble vs LightGBM solo ---
        le = LabelEncoder()
        yenc = le.fit_transform(y)
        splits = list(walk_forward(index, label_end, val["n_splits"], val["embargo_pct"],
                                   min_train=max(500, int(0.25 * len(dev)))))
        lgbm = lambda: LGBMClassifier(objective="multiclass", num_class=3, subsample_freq=1,
                                      random_state=42, verbosity=-1, **model_params)
        rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                            random_state=42, n_jobs=-1)
        gbm = lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03,
                                                 subsample=0.8, random_state=42)

        p_lgbm = oos_proba(lgbm, X, yenc, w, splits, le.classes_)
        p_rf = oos_proba(rf, X, yenc, w, splits, le.classes_)
        p_gbm = oos_proba(gbm, X, yenc, w, splits, le.classes_)
        p_ens = (p_lgbm + p_rf + p_gbm) / 3

        valid = p_ens.dropna().index
        eval_index = dev.index[(dev.index >= valid[0]) & (dev.index <= valid[-1])]
        df_eval = df.loc[eval_index]

        def to_signal(p, mode):
            pred = le.inverse_transform(p.loc[eval_index].to_numpy().argmax(axis=1))
            s = pd.Series(pred, index=eval_index)
            return (s > 0).astype(float) if mode == "lf" else s.astype(float)

        runs = {
            "LGBM long/flat": run_backtest(df_eval, to_signal(p_lgbm, "lf"), fee, slip, cap),
            "Ensemble long/flat": run_backtest(df_eval, to_signal(p_ens, "lf"), fee, slip, cap),
            "Ensemble long/short": run_backtest(df_eval, to_signal(p_ens, "ls"), fee, slip, cap),
        }
        print(f"\nJANSEN {tf}: modelo vs ensemble (OOS, neto)")
        print(pd.DataFrame([format_summary(n, summarize(bt, ppy)) for n, bt in runs.items()])
              .to_string(index=False))
        for name, p in (("LGBM", p_lgbm), ("Ensemble", p_ens)):
            pred = p.loc[eval_index].to_numpy().argmax(axis=1)
            acc = (le.inverse_transform(pred) == y.loc[eval_index]).mean()
            print(f"  accuracy {name}: {acc:.3f}")


if __name__ == "__main__":
    main()
