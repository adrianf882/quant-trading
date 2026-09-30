"""Descarga, guardado y validacion de datos OHLCV.

Se usa `ccxt` para bajar velas de un exchange (por defecto Binance) y
`pandas`/`pyarrow` para guardarlas en parquet.

Por que parquet: es columnar, comprime bien y preserva tipos y zona horaria,
a diferencia de un CSV.

Uso:
    py -m src.data            # descarga con los parametros de config.yaml
    py -m src.data --force    # vuelve a descargar aunque el archivo exista
"""
import argparse
import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from .config import get_paths, load_config

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Segundos por timeframe (los que vamos a usar). Evita instanciar el exchange
# solo para traducir el timeframe al calcular huecos.
TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                     "1h": 3600, "4h": 14400, "1d": 86400}


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"timeframe no soportado: {timeframe}")
    return pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])


def make_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """Crea el cliente del exchange con limite de velocidad activado.

    enableRateLimit evita que el exchange nos bloquee por hacer demasiadas
    peticiones seguidas (Binance aplica limites por minuto).
    """
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"El exchange '{exchange_id}' no existe en ccxt")
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    return exchange


def raw_data_path(symbol: str, timeframe: str) -> Path:
    """Ruta estandar del OHLCV crudo, parametrizada por simbolo (multi-activo)."""
    return get_paths()["raw"] / f"{symbol.replace('/', '_')}_{timeframe}.parquet"


def processed_path(symbol: str, timeframe: str, kind: str) -> Path:
    """Ruta de un dataset procesado (features, labels). Un archivo por simbolo."""
    return get_paths()["processed"] / f"{kind}_{symbol.replace('/', '_')}_{timeframe}.parquet"


def download_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: str = "2019-01-01T00:00:00Z",
    exchange_id: str = "binance",
    limit: int = 1000,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Descarga el historico completo de velas paginando por lotes.

    ccxt devuelve como maximo `limit` velas por llamada, asi que hay que
    pedir tramos avanzando el `since` hasta llegar al presente.
    """
    exchange = make_exchange(exchange_id)
    since_ms = exchange.parse8601(since)
    step_ms = exchange.parse_timeframe(timeframe) * 1000  # 1h -> 3.600.000 ms
    rows: list[list] = []

    while True:
        batch = None
        for attempt in range(max_retries):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as err:
                # Errores de red o del exchange: reintentamos con espera creciente.
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
                print(f"  reintento {attempt + 1} por: {err}", file=sys.stderr)

        if not batch:
            break

        rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + step_ms  # evitamos volver a pedir la ultima vela
        print(f"  {len(rows)} velas hasta {pd.to_datetime(last_ts, unit='ms', utc=True)}")

        if next_since <= since_ms:
            break  # sin progreso: cortamos para no quedar en loop infinito
        since_ms = next_since
        if len(batch) < limit:
            break  # ultimo tramo: ya no hay mas datos

    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df


def save_ohlcv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_ohlcv(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def gap_clusters(df: pd.DataFrame, timeframe: str = "1h") -> list[tuple]:
    """Agrupa las velas faltantes en intervalos contiguos (inicio, fin, n).

    Sirve para documentar los huecos: un corte de maintenance suele producir
    varias horas faltantes seguidas, no huecos aislados.
    """
    step = timeframe_to_timedelta(timeframe)
    expected = pd.date_range(df.index.min(), df.index.max(), freq=step, tz=df.index.tz)
    missing = expected.difference(df.index)
    if len(missing) == 0:
        return []
    clusters = []
    start = prev = missing[0]
    for ts in missing[1:]:
        if ts - prev == step:
            prev = ts
        else:
            clusters.append((start, prev, int((prev - start) / step) + 1))
            start = prev = ts
    clusters.append((start, prev, int((prev - start) / step) + 1))
    return clusters


def validate_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> dict:
    """Revisa integridad de la serie y devuelve un reporte.

    Chequea: orden temporal, duplicados, zona horaria, huecos, valores nulos,
    precios invalidos, consistencia OHLC y retornos extremos (posibles
    "quotes" erroneos, que Chan advierte que inflan los backtests).
    """
    report: dict = {}
    report["n_candles"] = int(len(df))
    report["start"] = df.index.min()
    report["end"] = df.index.max()
    report["tz"] = str(df.index.tz)
    report["sorted"] = bool(df.index.is_monotonic_increasing)
    report["duplicates"] = int(df.index.duplicated().sum())

    # Huecos: comparamos el indice real contra la rejilla teorica completa.
    step = timeframe_to_timedelta(timeframe)
    expected = pd.date_range(df.index.min(), df.index.max(), freq=step, tz=df.index.tz)
    missing = expected.difference(df.index)
    report["expected_candles"] = int(len(expected))
    report["n_gaps"] = int(len(missing))
    report["gap_examples"] = [str(ts) for ts in missing[:10]]
    report["gap_clusters"] = gap_clusters(df, timeframe)

    report["nan_counts"] = {c: int(df[c].isna().sum()) for c in OHLCV_COLUMNS}
    report["nonpositive_prices"] = int((df[["open", "high", "low", "close"]] <= 0).sum().sum())
    report["negative_volume"] = int((df["volume"] < 0).sum())

    # Consistencia OHLC: high debe ser el maximo y low el minimo.
    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    report["ohlc_inconsistent"] = int(bad_high + bad_low)

    # Retornos extremos: |log-ret| > 40% en una vela de 1h es sospechoso
    # (posible quote erroneo, que Chan advierte infla los backtests).
    log_ret = np.log(df["close"]).diff()
    report["extreme_returns"] = int((log_ret.abs() > 0.40).sum())
    return report


def print_report(report: dict, timeframe: str = "1h") -> None:
    print("\n" + "=" * 60)
    print("REPORTE DE VALIDACION DE DATOS")
    print("=" * 60)
    print(f"Velas:              {report['n_candles']:,}")
    print(f"Desde:              {report['start']}")
    print(f"Hasta:              {report['end']}")
    print(f"Zona horaria:       {report['tz']}")
    print(f"Ordenado:           {report['sorted']}")
    print(f"Duplicados:         {report['duplicates']}")
    print(f"Rejilla esperada:   {report['expected_candles']:,} velas ({timeframe})")
    print(f"Huecos:             {report['n_gaps']}")
    if report["n_gaps"]:
        print("  Detalle de huecos (inicio -> fin, horas):")
        for start, end, n in report["gap_clusters"]:
            print(f"    {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M}  ({n}h)")
    print(f"NaN por columna:    {report['nan_counts']}")
    print(f"Precios <= 0:       {report['nonpositive_prices']}")
    print(f"Volumen negativo:   {report['negative_volume']}")
    print(f"OHLC inconsist.:    {report['ohlc_inconsistent']}")
    print(f"Retornos extremos:  {report['extreme_returns']} (|log-ret| > 40% en 1 vela)")
    ok = (
        report["duplicates"] == 0
        and report["ohlc_inconsistent"] == 0
        and report["nonpositive_prices"] == 0
        and report["sorted"]
    )
    print(f"\nChequeos criticos:  {'OK' if ok else 'REVISAR'}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y valida OHLCV")
    parser.add_argument("--force", action="store_true", help="re-descargar aunque exista")
    args = parser.parse_args()

    cfg = load_config()
    paths = get_paths()
    data_cfg = cfg["data"]
    symbol, timeframe, since = data_cfg["symbol"], data_cfg["timeframe"], data_cfg["since"]
    exchange_id = data_cfg.get("exchange", "binance")

    safe_symbol = symbol.replace("/", "_")
    out_path = raw_data_path(symbol, timeframe)

    if out_path.exists() and not args.force:
        print(f"Usando datos ya descargados: {out_path}")
        df = load_ohlcv(out_path)
    else:
        print(f"Descargando {symbol} {timeframe} desde {since} en {exchange_id}...")
        df = download_ohlcv(symbol, timeframe, since, exchange_id)
        save_ohlcv(df, out_path)
        print(f"Guardado en {out_path}")

    report = validate_ohlcv(df, timeframe)
    print_report(report, timeframe)


if __name__ == "__main__":
    main()
