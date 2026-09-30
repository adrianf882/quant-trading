"""Modelo baseline (LightGBM) sobre features + labels.

Decisiones tomadas teniendo en cuenta a Lopez de Prado, Advances in Financial
Machine Learning (2018):

- El triple barrier vuelve las etiquetas *path-dependent* y solapadas (linea
  1892). Eso hace que las muestras no sean independientes, asi que las
  ponderamos por su **average uniqueness** (cap. 4, lineas 144 y 747).
- No alcanza con la accuracy: reportamos la **confusion matrix** y
  precision/recall/F1 por clase (lineas 2071 y 2140).
- La validacion seria (purged k-fold + embargo, lineas 3870 y 4006) llega en
  la Fase 6. Aca hacemos un split temporal simple y reservamos sin tocar el
  ~20% final (holdout de la Fase 8).
- La importanza MDI esta sesgada hacia features de alta cardinalidad (cap. 8);
  se reporta como preliminar, y la MDA/PFI con purga queda para mas adelante.

Uso:
    py -m src.models
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from .config import load_config
from .data import load_ohlcv, processed_path, raw_data_path
from .features import FEATURE_COLUMNS, build_features
from .labels import triple_barrier


def average_uniqueness(index: pd.DatetimeIndex, label_end: pd.Series) -> np.ndarray:
    """Uniqueness media de cada etiqueta (Prado, cap. 4).

    Concurrencia c(t) = cantidad de etiquetas activas en la vela t. La
    uniqueness de una etiqueta que abarca [i, end] es el promedio de 1/c(t)
    en ese tramo. Etiquetas muy solapadas pesan menos.
    """
    n = len(index)
    valid = label_end.notna().to_numpy()
    start_pos = np.where(valid)[0]
    end_pos = index.get_indexer(label_end[valid])
    # Si el fin de etiqueta no esta en el indice, tratamos la etiqueta como de 1 vela.
    missing = end_pos < 0
    if missing.any():
        end_pos = end_pos.copy()
        end_pos[missing] = start_pos[missing]

    diff = np.zeros(n + 1)
    np.add.at(diff, start_pos, 1)
    np.add.at(diff, end_pos + 1, -1)
    concurrency = np.cumsum(diff[:-1])
    concurrency[concurrency == 0] = 1  # velas sin etiqueta activa

    inv = 1.0 / concurrency
    prefix = np.concatenate([[0.0], np.cumsum(inv)])
    uniq = (prefix[end_pos + 1] - prefix[start_pos]) / (end_pos - start_pos + 1)

    out = np.full(n, np.nan)
    out[start_pos] = uniq
    return out


def build_dataset(features: pd.DataFrame, labels: pd.DataFrame,
                  feature_columns: list[str]) -> pd.DataFrame:
    """Une features y labels y descarta calentamiento / sin etiqueta."""
    data = features[feature_columns].copy()
    data["label"] = labels["label"]
    data["label_end"] = labels["label_end"]
    data["weight"] = average_uniqueness(labels.index, labels["label_end"])
    data = data.dropna(subset=feature_columns + ["label"])
    return data


def temporal_split(n: int, test_frac: float = 0.3):
    """Split temporal puro (nada de shuffle): train antes, test despues."""
    cut = int(n * (1 - test_frac))
    train = np.zeros(n, dtype=bool)
    train[:cut] = True
    return train, ~train


def make_model(params: dict) -> LGBMClassifier:
    # subsample_freq=1 es necesario para que `subsample` tenga efecto en LightGBM.
    model_params = dict(
        objective="multiclass",
        num_class=3,
        subsample_freq=1,
        importance_type="gain",
        random_state=42,
        verbosity=-1,
        **params,
    )
    return LGBMClassifier(**model_params)


def make_binary_model(params: dict) -> LGBMClassifier:
    """Modelo binario (para el meta-modelo de meta-labeling)."""
    model_params = dict(
        objective="binary",
        subsample_freq=1,
        importance_type="gain",
        random_state=42,
        verbosity=-1,
        **{k: v for k, v in params.items() if k != "num_class"},
    )
    return LGBMClassifier(**model_params)


def main() -> None:
    cfg = load_config()
    data_cfg, val_cfg, model_cfg = cfg["data"], cfg["validation"], cfg["model"]
    symbol, timeframe = data_cfg["symbol"], data_cfg["timeframe"]

    df = load_ohlcv(raw_data_path(symbol, timeframe))
    features = build_features(df)
    labels = triple_barrier(df, k=cfg["labels"]["k"], horizon=cfg["labels"]["horizon"],
                            vol_window=cfg["labels"]["vol_window"])
    data = build_dataset(features, labels, FEATURE_COLUMNS)

    # Reservamos el 20% final SIN TOCAR hasta la Fase 8.
    holdout_cut = int(len(data) * (1 - val_cfg["holdout_pct"]))
    dev = data.iloc[:holdout_cut]
    print(f"Dataset: {len(data):,} filas | dev (Fase 5): {len(dev):,} | "
          f"holdout intocable (Fase 8): {len(data) - holdout_cut:,}")

    train_mask, test_mask = temporal_split(len(dev), test_frac=0.3)
    X = dev[FEATURE_COLUMNS]
    y = dev["label"]
    w = dev["weight"]
    print(f"Train: {train_mask.sum():,} | Test: {test_mask.sum():,} (temporal, sin shuffle)")
    print(f"Rango test: {dev.index[test_mask][0]} -> {dev.index[test_mask][-1]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = make_model(model_cfg["params"])
    model.fit(X[train_mask], y_enc[train_mask], sample_weight=w[train_mask])

    pred = model.predict(X[test_mask])
    y_test = y_enc[test_mask]

    print("\nMatriz de confusion (filas=real, columnas=predicho):")
    cm = confusion_matrix(y_test, pred)
    print(pd.DataFrame(cm, index=[f"real {c}" for c in le.classes_],
                       columns=[f"pred {c}" for c in le.classes_]).to_string())
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, pred, target_names=[str(c) for c in le.classes_],
                                digits=3, zero_division=0))

    print("Importancia de features (MDI/gain; sesgada, preliminar):")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(imp.sort_values(ascending=False).to_string())

    acc = (pred == y_test).mean()
    print(f"\nAccuracy global: {acc:.3f}  (>= 0.333 = mejor que azar en 3 clases)")
    print("Nota: este split no purga el solapamiento de etiquetas; la validacion "
          "honesta es la Fase 6 (purged k-fold + embargo).")


if __name__ == "__main__":
    main()
