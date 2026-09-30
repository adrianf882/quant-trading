"""Tests de los esquemas de validacion temporal."""
import numpy as np
import pandas as pd

from src.validation import label_end_positions, purged_kfold, walk_forward


def make_labels(n=200, horizon=10):
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    end = np.minimum(np.arange(n) + horizon, n - 1)
    label_end = pd.Series(idx[end], index=idx)
    return idx, label_end


def test_purged_kfold_cubre_todo_y_no_solapa():
    idx, label_end = make_labels(200, horizon=10)
    n = len(idx)
    end_pos = label_end_positions(idx, label_end)
    covered = np.zeros(n, dtype=bool)

    for train_pos, test_pos in purged_kfold(idx, label_end, n_splits=5, embargo_pct=0.02):
        covered[test_pos] = True
        test_start, test_stop = test_pos[0], test_pos[-1]
        test_span_end = end_pos[test_pos].max()
        for i in train_pos:
            # Ninguna etiqueta del train puede solapar el tramo del test...
            assert not (i <= test_span_end and end_pos[i] >= test_start)
            # ...ni caer en el embargo posterior.
            assert not (test_stop < i < test_stop + int(n * 0.02))
    assert covered.all()


def test_walk_forward_train_siempre_antes_que_test():
    idx, label_end = make_labels(120, horizon=10)
    for train_pos, test_pos in walk_forward(idx, label_end, n_splits=3, min_train=40):
        assert train_pos.max() < test_pos.min()
