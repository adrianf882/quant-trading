"""Analisis por regimen: ganan o pierden las estrategias cuando el mercado NO es alcista.

Regimen causal (sin look-ahead):
  - alcista (bull): close > SMA200
  - no alcista: close <= SMA200
Dentro del no-alcista se distingue:
  - bajista: SMA200 con pendiente negativa
  - lateral: SMA200 con pendiente >= 0

Se atribuye el retorno de cada estrategia a cada regimen y se listan los
principales periodos no alcistas con el resultado de cada estrategia.

Uso:
    py -m src.regimes
"""
import numpy as np
import pandas as pd

from .config import get_paths, load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR
from .run_timeframes import build_runs


def compounded(r: pd.Series) -> float:
    return float((1 + r).prod() - 1) if len(r) else np.nan


def sharpe(r: pd.Series, ppy: int) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ppy)) if sd and not np.isnan(sd) else np.nan


def regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    sma200 = close.rolling(200).mean()
    slope = sma200.diff()
    out = pd.DataFrame(index=df.index)
    out["bull"] = close > sma200
    out["bear"] = (~out["bull"]) & (slope < 0)
    out["sideways"] = (~out["bull"]) & (slope >= 0)
    out.loc[sma200.isna(), ["bull", "bear", "sideways"]] = False
    return out


def nonbull_blocks(reg: pd.Series):
    """Devuelve los bloques contiguos no alcistas (inicio, fin)."""
    mask = (~reg).to_numpy()
    blocks = []
    start = None
    for i, is_nb in enumerate(mask):
        if is_nb and start is None:
            start = i
        elif not is_nb and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(mask) - 1))
    return blocks


def report_regime(runs: dict, df: pd.DataFrame, ppy: int, timeframe: str) -> None:
    common = runs["Buy & Hold"].index
    for bt in runs.values():
        common = common.intersection(bt.index)
    reg = regime_labels(df).reindex(common)
    bull = reg["bull"]

    print(f"\n{'='*70}\nREGIMEN {timeframe}: {bull.mean():.1%} del tiempo es alcista, "
          f"{1-bull.mean():.1%} no alcista\n{'='*70}")

    rows = []
    for name, bt in runs.items():
        r = bt.loc[common, "ret"]
        pos = bt.loc[common, "pos"]
        rb, rn = r[bull], r[~bull]
        rows.append({
            "estrategia": name,
            "ret_alcista": f"{compounded(rb):+.1%}",
            "ret_no_alcista": f"{compounded(rn):+.1%}",
            "sharpe_alcista": f"{sharpe(rb, ppy):.2f}",
            "sharpe_no_alcista": f"{sharpe(rn, ppy):.2f}",
            "exp_alcista": f"{(pos[bull] != 0).mean():.0%}",
            "exp_no_alcista": f"{(pos[~bull] != 0).mean():.0%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Desglose del no-alcista: bajista vs lateral.
    print(f"\nDesglose no-alcista ({timeframe}): bajista vs lateral")
    rows = []
    for name, bt in runs.items():
        r = bt.loc[common, "ret"]
        rb = r[reg["bear"]]
        rs = r[reg["sideways"]]
        rows.append({
            "estrategia": name,
            "ret_bajista": f"{compounded(rb):+.1%}",
            "ret_lateral": f"{compounded(rs):+.1%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Principales periodos no alcistas (>= 5 barras) con resultado por estrategia.
    blocks = [(s, e) for s, e in nonbull_blocks(bull) if e - s + 1 >= 5]
    blocks = sorted(blocks, key=lambda b: b[1] - b[0], reverse=True)[:8]
    print(f"\nPeriodos no alcistas mas largos ({timeframe}): retorno por estrategia")
    rows = []
    for s, e in blocks:
        idx = common[s:e + 1]
        row = {"periodo": f"{idx[0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}", "dias": len(idx)}
        for name in runs:
            row[name] = f"{compounded(runs[name].loc[idx, 'ret']):+.1%}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def analyze(df: pd.DataFrame, cfg: dict, ppy: int, timeframe: str,
            short_variants: bool = False) -> None:
    runs = build_runs(df, cfg, ppy, short_variants=short_variants)
    report_regime(runs, df, ppy, timeframe)


def main() -> None:
    cfg = load_config()
    symbol = cfg["data"]["symbol"]
    for tf in ("1d", "4h"):
        df = load_ohlcv(raw_data_path(symbol, tf))
        analyze(df, cfg, PERIODS_PER_YEAR[tf], tf)


if __name__ == "__main__":
    main()
