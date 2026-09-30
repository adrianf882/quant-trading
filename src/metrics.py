"""Metricas de performance para evaluar estrategias.

Todas las metricas de riesgo se anualizan usando `periods_per_year`
(para velas de 1h: 24 * 365 = 8760).

Sigue la recomendacion de Chan (Quantitative Trading, cap. 3) de reportar un
conjunto minimo de medidas estandar: retorno, Sharpe, drawdown y costos, y no
solo el retorno total.
"""
import numpy as np
import pandas as pd

# Periodos por ano segun timeframe (mercado cripto: 24/7).
PERIODS_PER_YEAR = {"1m": 525600, "5m": 105120, "15m": 35040, "30m": 17520,
                    "1h": 8760, "4h": 2190, "1d": 365}


def annualized_return(equity: pd.Series) -> float:
    """CAGR usando tiempo calendario (robusto a huecos)."""
    if len(equity) < 2:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    if years <= 0:
        return np.nan
    return (1 + total) ** (1 / years) - 1


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    sd = r.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    if downside == 0:
        return np.nan
    return float(r.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """Duracion del tramo mas largo por debajo de un maximo previo."""
    peak = equity.cummax()
    underwater = equity < peak
    longest = pd.Timedelta(0)
    start = prev = None
    for ts, uw in underwater.items():
        if uw:
            if start is None:
                start = ts
            prev = ts
        elif start is not None:
            longest = max(longest, prev - start)
            start = None
    if start is not None:
        longest = max(longest, prev - start)
    return longest


def extract_trades(returns: pd.Series, positions: pd.Series) -> list[float]:
    """Devuelve el retorno de cada operacion (tramo de posicion constante != 0)."""
    pos = positions.values
    ret = returns.values
    trades: list[float] = []
    current = None
    acc = 1.0
    for i in range(len(pos)):
        p = pos[i]
        if current is None or p != current:
            if current is not None and current != 0:
                trades.append(acc - 1)
            current = p
            acc = 1.0
        acc *= (1 + ret[i])
    if current is not None and current != 0:
        trades.append(acc - 1)
    return trades


def trade_log(returns: pd.Series, positions: pd.Series) -> pd.DataFrame:
    """Lista de operaciones con entrada, salida, duracion y retorno."""
    pos = positions.to_numpy()
    idx = positions.index
    ret = np.asarray(returns.to_numpy(), dtype=float)
    rows = []
    current = None
    acc = 1.0
    entry = None
    for i, p in enumerate(pos):
        if current is None or p != current:
            if current is not None and current != 0:
                rows.append({"entrada": entry, "salida": idx[i], "retorno": acc - 1})
            current = p
            acc = 1.0
            entry = idx[i]
        acc *= (1 + ret[i])
    if current is not None and current != 0:
        rows.append({"entrada": entry, "salida": idx[-1], "retorno": acc - 1})
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame, periods_per_year: int) -> dict:
    """Resume una corrida de backtest (columnas: ret, pos, equity)."""
    equity = bt["equity"]
    ret = bt["ret"]
    trades = extract_trades(ret, bt["pos"])
    arr = np.array(trades) if trades else np.array([0.0])
    wins = arr[arr > 0].sum()
    losses = arr[arr < 0].sum()
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "ann_return": float(annualized_return(equity)),
        "sharpe": sharpe(ret, periods_per_year),
        "sortino": sortino(ret, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "dd_duration_days": drawdown_duration(equity).total_seconds() / 86400,
        "n_trades": len(trades),
        "win_rate": float((arr > 0).mean()) if trades else np.nan,
        "profit_factor": float(wins / abs(losses)) if losses < 0 else np.nan,
        "exposure": float((bt["pos"] != 0).mean()),
    }


def format_summary(name: str, m: dict) -> dict:
    """Prepara una fila legible para la tabla comparativa."""
    return {
        "estrategia": name,
        "ret_total": f"{m['total_return']:.1%}",
        "ret_anual": f"{m['ann_return']:.1%}",
        "sharpe": f"{m['sharpe']:.2f}",
        "sortino": f"{m['sortino']:.2f}",
        "max_dd": f"{m['max_drawdown']:.1%}",
        "dd_dias": f"{m['dd_duration_days']:.0f}",
        "ops": m["n_trades"],
        "win_rate": f"{m['win_rate']:.1%}" if not np.isnan(m["win_rate"]) else "-",
        "profit_factor": f"{m['profit_factor']:.2f}" if not np.isnan(m["profit_factor"]) else "-",
        "exposicion": f"{m['exposure']:.1%}",
    }
