"""Motor de backtest vectorizado con costos.

Regla de alineacion temporal (la mas importante):
- Una senal generada con la informacion disponible al CIERRE de la vela `t`
  se ejecuta en la APERTURA de la vela `t+1`.
- La posicion se mantiene durante la vela siguiente y su retorno se mide
  open(t) -> open(t+1).

Chan (Quantitative Trading, cap. 3, linea 625) advierte exactamente este punto:
un ejemplo de look-ahead es "buy when the stock is within 1 percent of the
day's low", porque el minimo del dia no se conoce hasta el cierre.

Uso:
    py -m src.backtest        # corre los baselines contra los datos reales
"""
import numpy as np
import pandas as pd

from .config import load_config
from .baselines import buy_and_hold_signal, sma_cross_signal
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR, format_summary, summarize


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    fee: float = 0.001,
    slippage: float = 0.0005,
    initial_capital: float = 10000.0,
    price: str = "open",
) -> pd.DataFrame:
    """Corre el backtest y devuelve posiciones, retornos y curva de equity.

    Parametros
    ----------
    df : DataFrame con columnas OHLCV indexado por timestamp UTC.
    signal : posicion objetivo decidida al cierre de cada vela (0/1 o -1..1).
             Debe construirse usando SOLO datos hasta `t`.
    fee : comision por lado (0.1% por defecto).
    slippage : deslizamiento por lado (0.05% por defecto).
    price : columna de precio de ejecucion (por defecto "open").
    """
    # Ejecucion en la apertura de t+1: la posicion vigente en `t` es la senal de t-1.
    pos = signal.shift(1).fillna(0.0)

    # Retorno open(t) -> open(t+1). Es el retorno de mantener durante la vela t.
    ret = df[price].pct_change().shift(-1)

    # Costo por cambio de posicion, cobrado cuando ocurre la ejecucion.
    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * (fee + slippage)

    gross_ret = pos * ret
    strat_ret = (gross_ret - cost).fillna(0.0)
    equity = initial_capital * (1 + strat_ret).cumprod()

    out = pd.DataFrame(
        {
            "signal": signal,
            "pos": pos,
            "ret": strat_ret,
            "gross_ret": gross_ret.fillna(0.0),
            "cost": cost,
            "equity": equity,
        }
    )
    # La ultima vela no tiene "siguiente apertura": su retorno no es realizable.
    return out.iloc[:-1]


def main() -> None:
    cfg = load_config()
    data_cfg = cfg["data"]
    bt_cfg = cfg["backtest"]
    timeframe = data_cfg["timeframe"]
    ppy = PERIODS_PER_YEAR[timeframe]

    df = load_ohlcv(raw_data_path(data_cfg["symbol"], timeframe))

    common = dict(
        fee=bt_cfg["fee"],
        slippage=bt_cfg["slippage"],
        initial_capital=bt_cfg["initial_capital"],
    )

    runs = {
        "Buy & Hold": run_backtest(df, buy_and_hold_signal(df), **common),
        "SMA 20/50 (neto)": run_backtest(df, sma_cross_signal(df, 20, 50), **common),
        "SMA 20/50 (sin costos)": run_backtest(
            df, sma_cross_signal(df, 20, 50), fee=0.0, slippage=0.0,
            initial_capital=bt_cfg["initial_capital"],
        ),
    }

    rows = [format_summary(name, summarize(bt, ppy)) for name, bt in runs.items()]
    table = pd.DataFrame(rows)
    print(f"\nBacktest {data_cfg['symbol']} {timeframe} | "
          f"fee={bt_cfg['fee']:.4f} slippage={bt_cfg['slippage']:.4f}")
    print(f"Periodo: {df.index.min()} -> {df.index.max()}")
    print(table.to_string(index=False))

    cost_total = runs["SMA 20/50 (neto)"]["cost"].sum()
    print(f"\nCosto total acumulado SMA 20/50: {cost_total:.4f} (fraccion del capital)")
    print("Recorda: el objetivo es que el modelo supere a estos baselines DESPUES de costos.")


if __name__ == "__main__":
    main()
