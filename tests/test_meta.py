"""Tests de meta-labeling (etiquetas binarias y senal no solapada)."""
import numpy as np
import pandas as pd

from src.labels import meta_labels
from src.meta import meta_signal


def synth(n=40):
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    # Retornos log alternados (+-2%) -> volatilidad estable y no nula.
    alt = np.array([0.02, -0.02] * (n // 2 + 1))[:n]
    close = 100 * np.exp(np.cumsum(alt))
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
        index=idx,
    )
    return df


def test_meta_label_largo_acierta_con_subida():
    df = synth(40)
    side = pd.Series(1.0, index=df.index)  # siempre long
    df.iloc[8, df.columns.get_loc("high")] = 200.0  # gran subida en la vela 8
    res = meta_labels(df, side, k=1.0, horizon=10, vol_window=5)
    assert res["meta_label"].iloc[6] == 1.0  # entrada en 6, ganancia en 8


def test_meta_label_corto_acierta_con_bajada():
    df = synth(40)
    side = pd.Series(-1.0, index=df.index)  # siempre short
    df.iloc[8, df.columns.get_loc("low")] = 1.0  # gran bajada en la vela 8
    res = meta_labels(df, side, k=1.0, horizon=10, vol_window=5)
    assert res["meta_label"].iloc[6] == 1.0


def test_meta_signal_no_solapa():
    idx = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
    side = pd.Series([0, 1, 1, 1, 1, 0, -1, -1, 0, 0], index=idx, dtype=float)
    prob = pd.Series(0.9, index=idx)
    end = pd.Series([pd.NaT, idx[4], idx[4], idx[4], idx[4], pd.NaT,
                     idx[8], idx[8], pd.NaT, pd.NaT], index=idx)
    sig = meta_signal(side, prob, end, threshold=0.5)
    # Primera apuesta en la fila 1, mantenida hasta la 4 (inclusive).
    assert (sig.iloc[1:5] == 1.0).all()
    # Segunda apuesta en la fila 6 (no solapa con la primera), hasta la 8.
    assert (sig.iloc[6:9] == -1.0).all()
    assert sig.iloc[0] == 0.0 and sig.iloc[5] == 0.0 and sig.iloc[9] == 0.0
