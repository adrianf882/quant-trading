"""Genera un dashboard HTML autocontenido (3 billeteras / estrategias).

- Equidad acumulada por billetera, con ESCALA COMPARTIDA (mismo rango Y en los 3).
- Invertido / Liquidez por billetera (segun el tipo de estrategia).
- Responsive. Escribe docs/index.html (Chart.js). Listo para GitHub Pages.

Uso:
    py -m src.dashboard
"""
import json
from pathlib import Path

import pandas as pd

from .config import get_paths

# (archivo, etiqueta, capital inicial, tipo)
WALLETS = [
    ("paper_log_futures.csv", "B-Kalman (futuros 4h)", 1000.0, "futures"),
    ("paper_log_spot.csv", "SPOT trend 1d", 100.0, "spot"),
    ("paper_log_carry.csv", "Carry de funding (delta-neutral)", 1000.0, "carry"),
]


def accounting(df: pd.DataFrame, equity: float, kind: str):
    """Devuelve (invertido, liquidez) segun el tipo de estrategia."""
    if kind == "futures":
        gross = abs(df["pos_btc"].iloc[-1]) + abs(df["pos_eth"].iloc[-1])
        inv = gross * equity
        return inv, max(0.0, equity - inv)
    if kind == "spot":
        inv = df["pos_btc"].iloc[-1] * equity
        return inv, max(0.0, equity - inv)
    # carry: compromete todo el capital (spot + margen del perp)
    return equity, 0.0


def build_series(csv_path: Path, label: str, capital: float, kind: str) -> dict:
    empty = {"label": label, "capital": capital, "points": [], "status": "sin datos",
             "invested": None, "liquidity": None}
    if not csv_path.exists():
        return empty
    df = pd.read_csv(csv_path)
    if df.empty:
        return empty
    cum = df["equity"] / capital - 1
    pts = [{"t": str(df["bar"].iloc[i]), "v": round(float(cum.iloc[i]) * 100, 3)}
           for i in range(len(df))]
    last = float(df["equity"].iloc[-1])
    status = "QUIEBRA" if last <= 0 else f"{last:,.0f} USD ({cum.iloc[-1]*100:+.1f}%)"
    inv, liq = accounting(df, last, kind)
    return {"label": label, "capital": capital, "points": pts, "status": status,
            "invested": inv, "liquidity": liq}


HTML = """<!doctype html>
<html lang="es"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quant Trading - Billeteras (paper)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  *{{box-sizing:border-box}} body{{font-family:system-ui,Arial,sans-serif;background:#0f1117;
    color:#e6e6e6;margin:0;padding:20px}}
  h1{{font-size:20px;font-weight:600;margin:0 0 2px}} .sub{{color:#9aa0a6;font-size:13px;margin:0 0 16px}}
  .grid{{display:grid;grid-template-columns:1fr;gap:18px;max-width:920px;margin:0 auto}}
  .card{{background:#171a21;border:1px solid #262a33;border-radius:12px;padding:16px}}
  .card h2{{font-size:15px;margin:0 0 4px}} .meta{{color:#9aa0a6;font-size:12px;margin-bottom:10px;line-height:1.5}}
  .bad{{color:#f7768e;font-weight:700}} .ok{{color:#9ece6a;font-weight:700}}
  .chartbox{{position:relative;width:100%;height:260px}}
  @media(min-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body>
<h1>Paper Trading - billeteras independientes</h1>
<p class="sub">Generado: {gen} · Simulacion (sin dinero real) · escala Y comun a los 3 graficos.</p>
<div class="grid">{cards}</div>
<script>
const DATA = {data};
const YMIN = {ymin}, YMAX = {ymax};
function draw(i){{
  const d = DATA[i], el = document.getElementById('c'+i);
  if(!d.points.length){{ el.parentElement.insertAdjacentHTML('beforeend','<p class="meta">Sin datos aun.</p>'); return; }}
  new Chart(el, {{
    type:'line',
    data:{{labels:d.points.map(p=>p.t), datasets:[{{
      label:'retorno acumulado %', data:d.points.map(p=>p.v),
      borderColor: i===0?'#7aa2f7':(i===1?'#4ec9b0':'#e0af68'),
      backgroundColor:'transparent', borderWidth:2,
      pointRadius: d.points.length<80?3:0, pointBackgroundColor:'#e6e6e6', tension:.15}}]}},
    options:{{responsive:true, maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#e6e6e6'}}}},
        tooltip:{{callbacks:{{label:x=>x.parsed.y.toFixed(2)+'%'}}}}}},
      scales:{{x:{{ticks:{{color:'#9aa0a6',maxTicksLimit:8}}}},
        y:{{min:YMIN, max:YMAX, ticks:{{color:'#9aa0a6',callback:v=>v+'%'}}}}}}}}
  }});
}}
DATA.forEach((_,i)=>draw(i));
</script>
</body></html>"""


def main() -> None:
    exp = get_paths()["experiments"]
    series = [build_series(exp / f, lab, cap, kind) for f, lab, cap, kind in WALLETS]

    def fmt(v):
        return f"{v:,.0f} USD" if v is not None else "—"

    cards = "".join(
        f'<div class="card"><h2>{s["label"]}</h2>'
        f'<div class="meta">Capital inicial: {s["capital"]:,.0f} USD · '
        f'Equity: <span class="{"bad" if s["status"]=="QUIEBRA" else "ok"}">{s["status"]}</span><br>'
        f'Invertido: <b>{fmt(s["invested"])}</b> · Liquidez: <b>{fmt(s["liquidity"])}</b></div>'
        f'<div class="chartbox"><canvas id="c{i}"></canvas></div></div>'
        for i, s in enumerate(series))

    vals = [p["v"] for s in series for p in s["points"]]
    lo, hi = (min(vals), max(vals)) if vals else (-1.0, 1.0)
    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    ymin, ymax = round(lo - span * 0.1, 2), round(hi + span * 0.1, 2)

    out = get_paths()["root"] / "docs" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out.write_text(HTML.format(gen=gen, cards=cards, data=json.dumps(series),
                               ymin=ymin, ymax=ymax), encoding="utf-8")
    print(f"Dashboard: {out} | escala Y [{ymin}, {ymax}] | "
          + " | ".join(f"{s['label']}={len(s['points'])}pts" for s in series))


if __name__ == "__main__":
    main()
