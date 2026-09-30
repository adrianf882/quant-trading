"""Validacion temporal para series financieras.

Dos esquemas, ambos evitando fuga de informacion:

1) PURGED K-FOLD con EMBARGO (Lopez de Prado, AFML cap. 7, lineas 3870 y 4006):
   - Se purga del train toda observacion cuya etiqueta solape en el tiempo con
     el test (porque las etiquetas del triple barrier se extienden al futuro).
   - Se aplica un embargo a las observaciones inmediatamente posteriores al test,
     porque las features estan autocorrelacionadas.

2) WALK-FORWARD: ventana de entrenamiento expansiva; se testea el tramo
   siguiente. Mas parecido a operar en la vida real.

Uso:
    py -m src.validation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from .config import load_config
from .data import load_ohlcv, raw_data_path
from .features import FEATURE_COLUMNS, build_features
from .labels import triple_barrier
from .models import build_dataset, make_model


def label_end_positions(index: pd.DatetimeIndex, label_end: pd.Series) -> np.ndarray:
    """Posicion (entera) donde se resuelve cada etiqueta. NaT -> su propia fila."""
    pos = index.get_indexer(label_end)
    unresolved = pos < 0
    pos[unresolved] = np.arange(len(index))[unresolved]
    return pos


def purged_kfold(index, label_end, n_splits=5, embargo_pct=0.01):
    """Genera (train_pos, test_pos) con purga de solapamiento y embargo."""
    n = len(index)
    end_pos = label_end_positions(index, label_end)
    fold_size = n // n_splits
    embargo = int(n * embargo_pct)

    for k in range(n_splits):
        start = k * fold_size
        stop = n if k == n_splits - 1 else (k + 1) * fold_size
        test_pos = np.arange(start, stop)
        test_span_end = int(end_pos[start:stop].max())

        train = []
        for i in range(n):
            if start <= i < stop:
                continue  # es test
            if stop <= i < stop + embargo:
                continue  # embargo posterior al test
            if i < start and end_pos[i] >= start:
                continue  # su etiqueta se resuelve dentro del test -> purgar
            if i >= stop and i <= test_span_end:
                continue  # su feature cae dentro del horizonte del test -> purgar
            train.append(i)
        yield np.array(train, dtype=int), test_pos


def walk_forward(index, label_end, n_splits=5, embargo_pct=0.01, min_train=5000):
    """Train expansivo; test = tramo siguiente. Purga por solapamiento + embargo."""
    n = len(index)
    end_pos = label_end_positions(index, label_end)
    embargo = int(n * embargo_pct)
    block = (n - min_train) // n_splits

    for k in range(n_splits):
        start = min_train + k * block
        stop = n if k == n_splits - 1 else min_train + (k + 1) * block
        test_pos = np.arange(start, stop)

        # Train = [0, start) purgado: etiquetas resueltas antes del test y
        # descartando las ultimas `embargo` observaciones.
        train = np.array([i for i in range(start - embargo) if end_pos[i] < start], dtype=int)
        yield train, test_pos


def walk_forward_predictions(model_params: dict, X: pd.DataFrame, y: np.ndarray,
                             w: np.ndarray, splits) -> pd.Series:
    """Predicciones OUT-OF-SAMPLE con walk-forward (para backtestear).

    Cada tramo de test se predice con un modelo entrenado SOLO con el pasado.
    Devuelve una Serie alineada a X.index, con NaN fuera de los tramos de test.
    """
    preds = pd.Series(np.nan, index=X.index, dtype=float)
    for train_pos, test_pos in splits:
        model = make_model(model_params)
        model.fit(X.iloc[train_pos], y[train_pos], sample_weight=w[train_pos])
        preds.iloc[test_pos] = model.predict(X.iloc[test_pos])
    return preds


def cross_validate(model_params: dict, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
                   splits) -> pd.DataFrame:
    """Entrena y evalua por fold. Devuelve metricas por fold."""
    rows = []
    for k, (train_pos, test_pos) in enumerate(splits):
        model = make_model(model_params)
        model.fit(X.iloc[train_pos], y[train_pos], sample_weight=w[train_pos])
        pred = model.predict(X.iloc[test_pos])
        rows.append({
            "fold": k,
            "n_train": len(train_pos),
            "n_test": len(test_pos),
            "accuracy": float((pred == y[test_pos]).mean()),
            "f1_macro": float(f1_score(y[test_pos], pred, average="macro")),
            "test_desde": X.index[test_pos[0]],
            "test_hasta": X.index[test_pos[-1]],
        })
    return pd.DataFrame(rows)


def _summary(name: str, df: pd.DataFrame) -> dict:
    return {
        "esquema": name,
        "accuracy (media)": f"{df['accuracy'].mean():.3f}",
        "accuracy (std)": f"{df['accuracy'].std():.3f}",
        "f1_macro (media)": f"{df['f1_macro'].mean():.3f}",
        "f1_macro (std)": f"{df['f1_macro'].std():.3f}",
    }


def main() -> None:
    cfg = load_config()
    data_cfg, val_cfg, model_cfg = cfg["data"], cfg["validation"], cfg["model"]
    symbol, timeframe = data_cfg["symbol"], data_cfg["timeframe"]

    df = load_ohlcv(raw_data_path(symbol, timeframe))
    features = build_features(df)
    labels = triple_barrier(df, k=cfg["labels"]["k"], horizon=cfg["labels"]["horizon"],
                            vol_window=cfg["labels"]["vol_window"])
    data = build_dataset(features, labels, FEATURE_COLUMNS)

    # Reservamos el 20% final sin tocar (Fase 8).
    dev = data.iloc[: int(len(data) * (1 - val_cfg["holdout_pct"]))]
    X = dev[FEATURE_COLUMNS]
    y = dev["label"].to_numpy()
    w = dev["weight"].to_numpy()
    index = dev.index
    label_end = dev["label_end"]

    n_splits, embargo_pct = val_cfg["n_splits"], val_cfg["embargo_pct"]
    params = model_cfg["params"]

    results = {}
    results["Purged K-Fold"] = cross_validate(
        params, X, y, w, purged_kfold(index, label_end, n_splits, embargo_pct))
    results["Walk-Forward"] = cross_validate(
        params, X, y, w, walk_forward(index, label_end, n_splits, embargo_pct))

    # Contraste: KFold aleatorio (MAL, solo para mostrar el sesgo optimista).
    rng = np.random.default_rng(0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    results["KFold aleatorio (incorrecto)"] = cross_validate(
        params, X, y, w, ((tr, te) for tr, te in kf.split(X)))

    for name, dfm in results.items():
        print(f"\n=== {name} : metricas por fold ===")
        show = dfm.copy()
        show["test_desde"] = show["test_desde"].dt.strftime("%Y-%m-%d")
        show["test_hasta"] = show["test_hasta"].dt.strftime("%Y-%m-%d")
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    summary = pd.DataFrame([_summary(n, d) for n, d in results.items()])
    print("\n=== Resumen comparativo ===")
    print(summary.to_string(index=False))
    print("\nRecordar: el KFold aleatorio infla las metricas (fuga por etiquetas "
          "solapadas y autocorrelacion). Los esquemas validos son Purged K-Fold y Walk-Forward.")


if __name__ == "__main__":
    main()
