"""Tests del par trading."""
import numpy as np
import pandas as pd
import pytest

from src.pairs import pair_positions, rolling_hedge_ratio, run_pair_backtest


def test_hedge_ratio_recupera_beta_conocida():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, 400)) + 4.0)
    y = 1.5 * x + rng.normal(0, 0.001, 400)  # beta real = 1.5
    beta = rolling_hedge_ratio(y, x, 100)
    assert abs(beta.iloc[-1] - 1.5) < 0.1


def test_posiciones_son_beta_neutrales():
    z = pd.Series([0, 3, 3, 3, 0], index=pd.RangeIndex(5), dtype=float)
    beta = pd.Series(2.0, index=z.index)
    sy, sx = pair_positions(z, beta, entry=2.0, exit=0.5)
    # Entra en t=1 (z>2) short spread: pos_y<0, pos_x>0.
    assert sy.iloc[1] < 0 and sx.iloc[1] > 0
    # Neutralidad beta: pos_x = -beta * pos_y  (sensibilidad neta a ETH = 0).
    assert abs(sx.iloc[1] + beta.iloc[1] * sy.iloc[1]) < 1e-9
    # Sale en t=4 (z<=0.5 vuelve a 0).
    assert sy.iloc[4] == 0 and sx.iloc[4] == 0


def test_backtest_suma_ambas_patas():
    idx = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
    dy = pd.DataFrame({"open": [100, 110, 121, 133.1, 146.41]}, index=idx)
    dx = pd.DataFrame({"open": [100, 100, 100, 100, 100]}, index=idx)
    sy = pd.Series(1.0, index=idx)   # long BTC
    sx = pd.Series(0.0, index=idx)
    bt = run_pair_backtest(dy, dx, sy, sx, fee=0.0, slippage=0.0, initial_capital=100.0)
    # +10% por barra durante 3 barras ejecutables -> equity ~ 100*1.1^3.
    assert bt["equity"].iloc[-1] == pytest.approx(100 * 1.1 ** 3)
