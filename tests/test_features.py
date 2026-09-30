"""Tests de features: verifican que NO haya look-ahead.

Metodo de Chan (Quantitative Trading, cap. 3): si una feature en `t` depende
solo del pasado, entonces construirla con todos los datos o con datos truncados
debe dar identicos resultados en la parte comun. Tambien probamos perturbando
el futuro.
"""
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features


def synth(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    ret = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.uniform(1, 10, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_features_no_dependen_del_futuro_truncando():
    df = synth(800)
    full = build_features(df)
    truncated = build_features(df.iloc[:500])
    pd.testing.assert_frame_equal(
        full.iloc[:500][FEATURE_COLUMNS],
        truncated[FEATURE_COLUMNS],
    )


def test_features_inmunes_a_perturbacion_del_futuro():
    df = synth(800, seed=11)
    base = build_features(df)

    # Perturbamos SOLO los precios de las ultimas 100 velas.
    modified = df.copy()
    modified.iloc[-100:, modified.columns.get_loc("close")] *= 1.5
    modified.iloc[-100:, modified.columns.get_loc("high")] *= 1.5
    perturbed = build_features(modified)

    # Las features hasta la vela 700 (exclusive) deben ser identicas.
    cut = len(df) - 100
    pd.testing.assert_frame_equal(
        base.iloc[:cut][FEATURE_COLUMNS],
        perturbed.iloc[:cut][FEATURE_COLUMNS],
    )
