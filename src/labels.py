"""Etiquetado con triple barrier (Lopez de Prado, Advances in Financial ML).

Para cada vela `t` se definen dos barreras horizontales a partir de la
volatilidad, y una barrera vertical (tiempo):

    sigma_h = vol_por_vela * sqrt(horizon)     # vol acumulada del horizonte
    barrera_sup = close_t * (1 + k * sigma_h)
    barrera_inf = close_t * (1 - k * sigma_h)
    barrera_vertical = t + horizon velas

Etiqueta:  1 si toca la superior primero, -1 si toca la inferior, 0 si vence.

Decisiones de diseno (para evitar data-snooping, ver Chan cap. 3 y 7):
- Las barreras escalan con sqrt(horizon): usar `k * vol_por_vela` sin escalar
  daria barreras tan estrechas que casi toda etiqueta se resuelve en 1-2 velas.
- Si en una MISMA vela se tocan ambas barreras, no se puede saber el orden con
  OHLC: asumimos la adversa (-1), que es la suposicion conservadora.

NOTA: a diferencia de las features, las etiquetas SI miran al futuro por
diseno (describen lo que paso despues de `t`). El test verifica que no miran
mas alla de `t + horizon`.
"""
import numpy as np
import pandas as pd

from .config import load_config
from .data import load_ohlcv, processed_path, raw_data_path


def triple_barrier(
    df: pd.DataFrame,
    k: float = 1.0,
    horizon: int = 24,
    vol_window: int = 24,
) -> pd.DataFrame:
    """Devuelve label {-1,0,1}, realized_ret y label_end (cuando se resolvio)."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = np.log(df["close"]).diff().rolling(vol_window).std().to_numpy()
    n = len(df)
    sqrt_h = np.sqrt(horizon)

    label = np.full(n, np.nan)
    realized = np.full(n, np.nan)
    end_idx = np.full(n, -1, dtype=np.int64)

    for i in range(n - horizon):
        v = vol[i]
        if not np.isfinite(v) or v <= 0:
            continue  # sin volatilidad estimable aun
        entry = close[i]
        up = entry * (1 + k * v * sqrt_h)
        down = entry * (1 - k * v * sqrt_h)

        lab, ret, end = 0.0, close[i + horizon] / entry - 1.0, i + horizon
        for j in range(i + 1, i + 1 + horizon):
            hit_up = high[j] >= up
            hit_down = low[j] <= down
            if hit_up and hit_down:
                lab, ret, end = -1.0, down / entry - 1.0, j  # conservador
                break
            if hit_up:
                lab, ret, end = 1.0, up / entry - 1.0, j
                break
            if hit_down:
                lab, ret, end = -1.0, down / entry - 1.0, j
                break

        label[i] = lab
        realized[i] = ret
        end_idx[i] = end

    end_ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    resolved = end_idx >= 0
    end_ts.iloc[np.where(resolved)[0]] = list(df.index[end_idx[resolved]])

    return pd.DataFrame(
        {"label": label, "realized_ret": realized, "label_end": end_ts},
        index=df.index,
    )


def meta_labels(
    df: pd.DataFrame,
    side: pd.Series,
    k: float = 1.0,
    horizon: int = 24,
    vol_window: int = 24,
) -> pd.DataFrame:
    """Etiquetas binarias para meta-labeling (Lopez de Prado, snippet 3.6).

    Dado el LADO de la apuesta (`side`: -1 o +1) decidido por un modelo
    primario, la etiqueta es 1 si la apuesta alcanza la barrera de ganancia
    antes que la de perdida, y 0 en caso contrario (stop o vencimiento).

    Donde `side == 0` no hay apuesta -> etiqueta NaN.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = np.log(df["close"]).diff().rolling(vol_window).std().to_numpy()
    side_arr = side.to_numpy(dtype=float)
    n = len(df)
    sqrt_h = np.sqrt(horizon)

    label = np.full(n, np.nan)
    end_idx = np.full(n, -1, dtype=np.int64)

    for i in range(n - horizon):
        s = side_arr[i]
        v = vol[i]
        if s == 0 or not np.isfinite(s) or not np.isfinite(v) or v <= 0:
            continue
        entry = close[i]
        up = entry * (1 + k * v * sqrt_h)    # barrera de ganancia si side=+1
        down = entry * (1 - k * v * sqrt_h)  # barrera de ganancia si side=-1

        lab, end = 0.0, i + horizon
        for j in range(i + 1, i + 1 + horizon):
            hit_up = high[j] >= up
            hit_down = low[j] <= down
            if hit_up and hit_down:
                lab = 0.0  # conservador: si se tocan ambas, asumimos stop
                end = j
                break
            if s > 0 and hit_up:
                lab, end = 1.0, j
                break
            if s < 0 and hit_down:
                lab, end = 1.0, j
                break
            if hit_up or hit_down:
                lab, end = 0.0, j
                break
        label[i] = lab
        end_idx[i] = end

    end_ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    resolved = end_idx >= 0
    end_ts.iloc[np.where(resolved)[0]] = list(df.index[end_idx[resolved]])
    return pd.DataFrame({"meta_label": label, "meta_end": end_ts}, index=df.index)


def main() -> None:
    cfg = load_config()
    data_cfg, lab_cfg = cfg["data"], cfg["labels"]
    symbol, timeframe = data_cfg["symbol"], data_cfg["timeframe"]

    df = load_ohlcv(raw_data_path(symbol, timeframe))
    res = triple_barrier(
        df,
        k=lab_cfg["k"],
        horizon=lab_cfg["horizon"],
        vol_window=lab_cfg["vol_window"],
    )

    out_path = processed_path(symbol, timeframe, "labels")
    res.to_parquet(out_path)

    valid = res["label"].dropna()
    total = len(valid)
    print(f"Triple barrier | k={lab_cfg['k']} horizon={lab_cfg['horizon']} "
          f"vol_window={lab_cfg['vol_window']}")
    print(f"Etiquetas validas: {total:,} | sin etiqueta: {res['label'].isna().sum():,}")
    print("\nDistribucion de etiquetas:")
    for v in (-1.0, 0.0, 1.0):
        c = int((valid == v).sum())
        print(f"  {int(v):>2}: {c:7,}  ({c / total:.1%})")

    print(f"\nRetorno medio por etiqueta (realized_ret):")
    for v in (-1.0, 0.0, 1.0):
        print(f"  {int(v):>2}: {res.loc[res['label'] == v, 'realized_ret'].mean():.4%}")

    print("\nRevision a mano (primeras 5 resueltas):")
    sample = res.dropna(subset=["label"]).head(5)
    for ts, row in sample.iterrows():
        print(f"  {ts:%Y-%m-%d %H:%M} close={df.loc[ts, 'close']:8.1f} "
              f"label={int(row['label']):>2} fin={row['label_end']:%Y-%m-%d %H:%M} "
              f"ret={row['realized_ret']:+.3%}")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
