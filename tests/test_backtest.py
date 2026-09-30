"""Tests del motor de backtest.

El test clave es `test_sin_lookahead_con_datos_truncados`, que implementa el
metodo de Chan (Quantitative Trading, cap. 3): correr la estrategia sobre todos
los datos y sobre datos truncados; las senales/posiciones en la parte comun
deben ser identicas. Si difieren, hay look-ahead.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.baselines import buy_and_hold_signal, sma_cross_signal


def synth(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Serie OHLCV sintetica reproducible (random walk lognormal)."""
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


def test_posicion_se_ejecuta_en_la_vela_siguiente():
    df = synth(20)
    signal = pd.Series(np.tile([0, 1], 10), index=df.index, dtype=float)
    bt = run_backtest(df, signal)
    expected = signal.shift(1).fillna(0.0).iloc[:-1]
    pd.testing.assert_series_equal(bt["pos"], expected, check_names=False)


def test_sin_lookahead_con_datos_truncados():
    df = synth(500, seed=1)
    full = sma_cross_signal(df, 20, 50)
    truncated = sma_cross_signal(df.iloc[:300], 20, 50)
    pd.testing.assert_series_equal(
        full.iloc[:300], truncated, check_names=False
    )


def test_costos_reducen_el_capital():
    df = synth(500, seed=2)
    signal = sma_cross_signal(df, 20, 50)
    sin_costos = run_backtest(df, signal, fee=0.0, slippage=0.0)
    con_costos = run_backtest(df, signal, fee=0.01, slippage=0.01)
    assert con_costos["equity"].iloc[-1] < sin_costos["equity"].iloc[-1]
    assert con_costos["cost"].sum() > 0


def test_buy_and_hold_sigue_al_precio():
    df = synth(300, seed=3)
    capital = 1000.0
    bt = run_backtest(df, buy_and_hold_signal(df), fee=0.0, slippage=0.0,
                      initial_capital=capital)
    # Sin costos, buy&hold debe reproducir open[-1] / open[1] (telescopa).
    expected = capital * df["open"].iloc[-1] / df["open"].iloc[1]
    assert bt["equity"].iloc[-1] == pytest.approx(expected, rel=1e-9)
