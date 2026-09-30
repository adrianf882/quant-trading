"""Tests de etiquetas (triple barrier).

Las etiquetas miran al futuro por diseno, pero SOLO hasta `t + horizon`.
Estos tests verifican esa causalidad y la regla conservadora when ambas
barreras se tocan en la misma vela.
"""
import numpy as np
import pandas as pd

from src.labels import triple_barrier


def synth(n: int = 300, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    ret = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.uniform(1, 10, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_etiquetas_no_miran_mas_alla_del_horizonte():
    horizon = 12
    df = synth(300, seed=5)
    base = triple_barrier(df, k=1.0, horizon=horizon, vol_window=24)

    # Perturbamos fuerte TODO lo posterior a la vela 200.
    cut = 200
    modified = df.copy()
    for col in ("open", "high", "low", "close"):
        modified.iloc[cut:, modified.columns.get_loc(col)] *= 3.0
    perturbed = triple_barrier(modified, k=1.0, horizon=horizon, vol_window=24)

    # Las etiquetas de t <= cut - horizon - 1 solo dependen de datos <= t+horizon < cut.
    safe = base.iloc[: cut - horizon - 1]
    safe_pert = perturbed.iloc[: cut - horizon - 1]
    pd.testing.assert_series_equal(safe["label"], safe_pert["label"])
    pd.testing.assert_series_equal(safe["realized_ret"], safe_pert["realized_ret"])


def test_ambas_barreras_misma_vela_es_conservador():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.arange(n) * 0.05  # tendencia suave -> vol > 0
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    # En la vela 12 forzamos el toque de AMBAS barreras a la vez.
    entry_i = 8
    horizon = 10
    df.iloc[12, df.columns.get_loc("high")] = 1000.0
    df.iloc[12, df.columns.get_loc("low")] = 1.0

    res = triple_barrier(df, k=1.0, horizon=horizon, vol_window=5)
    assert res["label"].iloc[entry_i] == -1.0


def test_resolucion_dentro_del_horizonte():
    horizon = 12
    df = synth(300, seed=9)
    res = triple_barrier(df, k=1.0, horizon=horizon, vol_window=24)
    resolved = res.dropna(subset=["label"])
    for ts, row in resolved.iterrows():
        pos = df.index.get_loc(ts)
        assert df.index[pos] < row["label_end"] <= df.index[pos + horizon]
