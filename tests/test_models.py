"""Tests de utilidades de modelado (uniqueness y split temporal)."""
import numpy as np
import pandas as pd
import pytest

from src.models import average_uniqueness, temporal_split


def test_uniqueness_con_etiquetas_solapadas():
    idx = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
    # Etiquetas en las filas 0 y 1, ambas resueltas en la vela 2.
    # La de la fila 0 abarca [0,2]; la de la fila 1 abarca [1,2].
    label_end = pd.Series([idx[2], idx[2], pd.NaT, pd.NaT, pd.NaT], index=idx)
    u = average_uniqueness(idx, label_end)
    # Concurrencia por vela: [1, 2, 2, 0, 0] -> uniqueness fila0 = (1+.5+.5)/3.
    assert u[0] == pytest.approx(2 / 3)
    assert u[1] == pytest.approx(0.5)
    assert np.isnan(u[2])


def test_split_temporal_es_contiguo():
    train, test = temporal_split(100, test_frac=0.3)
    assert train.sum() == 70 and test.sum() == 30
    assert not (train & test).any()
    assert train[:70].all() and test[70:].all()
