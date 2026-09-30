"""Experimento Prado: dollar bars + diferenciacion fraccionaria.

Compara tres variantes sobre BTC:
  1) barras de tiempo 1d (referencia),
  2) dollar bars (sampleo por actividad),
  3) dollar bars + features con diferenciacion fraccionaria (memoria).

La evaluacion es por criterio ABSOLUTO (retorno, Sharpe, drawdown), no por
ganarle a buy & hold. Uso:
    py -m src.run_bars
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .bars import build_dollar_bars
from .config import get_paths, load_config
from .data import load_ohlcv, raw_data_path
from .features import build_features
from .fracdiff import add_fracdiff_features
from .metrics import format_summary
from .run_timeframes import run_all


def bars_per_year(bars: pd.DataFrame) -> int:
    years = (bars.index[-1] - bars.index[0]).days / 365.25
    return int(round(len(bars) / years))


def main() -> None:
    cfg = load_config()
    symbol = cfg["data"]["symbol"]

    daily = load_ohlcv(raw_data_path(symbol, "1d"))
    dbars = build_dollar_bars(load_ohlcv(raw_data_path(symbol, "5m")),
                              target_minutes=60, window=288, source_timeframe="5m")
    print(f"1d: {len(daily):,} velas | dollar bars: {len(dbars):,} "
          f"(~{bars_per_year(dbars)} por ano)")

    variants = {
        "1d (tiempo)": (daily, bars_per_year(daily), None),
        "Dollar bars": (dbars, bars_per_year(dbars), None),
    }
    # Variante con fracdiff (calcula d por ADF sobre cada serie).
    fd = add_fracdiff_features(build_features(dbars), dbars)
    print(f"Fracdiff: d_close={fd.attrs['d_close']:.2f} d_volume={fd.attrs['d_volume']:.2f}")
    variants["Dollar + fracdiff"] = (dbars, bars_per_year(dbars), fd)

    all_rows = []
    for name, (df, ppy, features) in variants.items():
        print(f"\n=== {name} ===")
        summaries = run_all(df, cfg, ppy, features=features)
        for strat, m in summaries.items():
            all_rows.append({"variante": name, **format_summary(strat, m)})
        print(pd.DataFrame([format_summary(s, m) for s, m in summaries.items()]).to_string(index=False))

    full = pd.DataFrame(all_rows)
    exp = get_paths()["experiments"]
    exp.mkdir(parents=True, exist_ok=True)
    full.to_csv(exp / "dollar_fracdiff_summary.csv", index=False)
    with open(exp / "dollar_fracdiff_summary.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "rows": all_rows}, f, indent=2, ensure_ascii=False)
    print("\n=== RESUMEN (criterio absoluto) ===")
    print(full.to_string(index=False))
    print(f"\nGuardado en {exp / 'dollar_fracdiff_summary.csv'}")


if __name__ == "__main__":
    main()
