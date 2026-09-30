"""Variante con SHORT: evalua long/short en regimenes alcistas y no alcistas.

Incluye: ML long/short (clasificador 3 clases tal cual), SMA long/short, y
meta-labeling con primario long/short (side = +1 si close>SMA200, -1 si no).

ADVERTENCIA: el short en cripto requiere perpetuos/margen -> existe un costo de
funding que NO se modela aca. Los resultados short son por lo tanto optimistas.

Uso:
    py -m src.run_short
"""
from .config import load_config
from .data import load_ohlcv, raw_data_path
from .metrics import PERIODS_PER_YEAR
from .regimes import analyze


def main() -> None:
    cfg = load_config()
    symbol = cfg["data"]["symbol"]
    for tf in ("1d", "4h"):
        print(f"\n\n############ SHORT VARIANTS - {tf} ############")
        df = load_ohlcv(raw_data_path(symbol, tf))
        analyze(df, cfg, PERIODS_PER_YEAR[tf], tf, short_variants=True)


if __name__ == "__main__":
    main()
