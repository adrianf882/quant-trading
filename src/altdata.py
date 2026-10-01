"""Datos alternativos: sentimiento (Fear & Greed) y Google Trends.

- Fear & Greed Index (alternative.me): diario, 2018+, sin auth. Robusto.
- Google Trends (pytrends): interes relativo 0-100 por termino. Fragil.

Shim: pytrends es incompatible con urllib3 2.x (usa `method_whitelist`); se
parchea Retry para aceptar el alias y no romper.

Uso:
    py -m src.altdata
"""
import urllib3.util.retry as _retry

import pandas as pd
import requests

from .config import get_paths
from .data import save_ohlcv

_orig_retry_init = _retry.Retry.__init__


def _patched_retry(self, *args, method_whitelist=None, allowed_methods=None, **kwargs):
    if allowed_methods is None:
        allowed_methods = method_whitelist
    return _orig_retry_init(self, *args, allowed_methods=allowed_methods, **kwargs)


_retry.Retry.__init__ = _patched_retry


def fetch_fear_greed() -> pd.DataFrame:
    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.set_index("date").sort_index()
    df["fng"] = df["value"].astype(float)
    return df[["fng"]]


def fetch_google_trends(terms, timeframe="2019-01-01 2026-09-30") -> pd.DataFrame:
    from pytrends.request import TrendReq
    p = TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=0.5)
    p.build_payload(list(terms), timeframe=timeframe)
    df = p.interest_over_time()
    if "isPartial" in df:
        df = df.drop(columns=["isPartial"])
    return df


def fetch_gtrends_weekly(terms, chunks=(("2019-01-01", "2023-06-30"),
                                       ("2023-01-01", "2026-09-30"))) -> pd.DataFrame:
    """Google Trends semanal: pide por chunks (<=5 anos dan semanal) y los une
    reescalando el solapamiento para empalmar la normalizacion."""
    import time
    base = None
    for start, end in chunks:
        df = fetch_google_trends(terms, f"{start} {end}")
        time.sleep(1)
        if base is None:
            base = df
            continue
        ov = base.index.intersection(df.index)
        ratio = (base.loc[ov].mean() / df.loc[ov].mean()) if len(ov) else 1.0
        df = df * ratio
        base = pd.concat([base[~base.index.isin(df.index)], df]).sort_index()
    return base



def main() -> None:
    paths = get_paths()
    raw = paths["raw"]

    fng = fetch_fear_greed()
    save_ohlcv(fng, raw / "fng.parquet")
    print(f"Fear & Greed: {len(fng)} dias | {fng.index.min().date()} -> {fng.index.max().date()} | "
          f"rango {fng['fng'].min():.0f}-{fng['fng'].max():.0f}")

    try:
        gt = fetch_gtrends_weekly(["bitcoin", "buy bitcoin", "bitcoin crash"])
        save_ohlcv(gt, raw / "gtrends.parquet")
        step = gt.index[1] - gt.index[0]
        print(f"Google Trends semanal: {len(gt)} filas | paso ~{step} | "
              f"{gt.index.min().date()} -> {gt.index.max().date()}")
    except Exception as err:  # noqa: BLE001
        print(f"Google Trends fallo (esperable/fragil): {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
