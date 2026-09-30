"""Estrategias baseline (sin ML), para tener un piso a vencer.

Ambas devuelven la posicion objetivo al cierre de cada vela, calculada con
informacion disponible hasta esa vela (sin look-ahead). El motor de backtest
se encarga de ejecutar en la apertura siguiente.
"""
import pandas as pd


def buy_and_hold_signal(df: pd.DataFrame) -> pd.Series:
    """Siempre en mercado. Es el benchmark mas duro en un activo alcista."""
    return pd.Series(1.0, index=df.index, name="buy_hold")


def sma_cross_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long cuando la media movil rapida esta por encima de la lenta.

    Cruce de medias 20/50: la senal usa el cierre hasta `t` inclusive, asi que
    se puede ejecutar en `t+1`. Antes de tener `slow` velas quedamos afuera.
    """
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float)
    signal[slow_ma.isna()] = 0.0
    return signal.rename("sma_cross")
