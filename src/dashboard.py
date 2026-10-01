"""Genera un dashboard HTML autocontenido (3 billeteras / estrategias).

Cada linea tiene su PROPIO capital (billetera) y se muestra el retorno acumulado
+ si "quiebra" (equity <= 0). Escribe docs/index.html (Chart.js). Listo para Pages.

Uso:
    py -m src.dashboard
"""
import json
from pathlib import Path

import pandas as pd

from .config import get_paths

# (archivo, etiqueta, capital inicial de la billetera)
WALLETS = [
    ("paper_log_futures.csv", "B-Kalman (futuros 4h)", 1000.0),
    ("paper_log_spot.csv", "SPOT trend 1d", 100.0),
    ("paper_log_carry.csv", "Carry de funding (delta-neutral)", 1000.0),
]


def build_series(csv_path: Path, label: str, capital: float) -> dict:
    if not csv_path.exists():
        return {"label": label, "capital": capital, "points": [], "status": "sin datos"}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"label": label, "capital": capital, "points": [], "status": "sin datos"}
    cum = df["equity"] / capital - 1
    pts = [{"t": str(df["bar"].iloc[i]), "v": round(float(cum.iloc[i]) * 100, 3),
            "eq": round(float(df["equity"].iloc[i]), 2)} for i in range(len(df))]
    last = float(df["equity"].iloc[-1])
    status = "QUIEBRA" if last <= 0 else f"{last:,.0f} USD ({cum.iloc[-1]*100:+.1f}%)"
    return {"label": label, "capital": capital, "points": pts, "status": status}


HTML = """<!doctype html>
<html lang="es"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quant Trading - Billeteras (paper)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body{{font-family:system-ui,Arial,sans-serif;background:#0f1117;color:#e6e6e6;margin:0;padding:24px}}
  h1{{font-size:20px;font-weight:600}} .sub{{color:#9aa0a6;font-size:13px;margin-top:-8px}}
  .grid{{display:grid;grid-template-columns:1fr;gap:20px;max-width:900px;margin:auto}}
  .card{{background:#171a21;border:1px solid #262a33;border-radius:12px;padding:16px}}
  .card h2{{font-size:15px;margin:0 0 4px}} .meta{{color:#9aa0a6;font-size:12px;margin-bottom:10px}}
  .bad{{color:#f7768e;font-weight:700}} .ok{{color:#9ece6a;font-weight:700}}
  canvas{{max-height:300px}}
</style></head>
<body>
<h1>Paper Trading - billeteras independientes</h1>
<p class="sub">Generado: {gen} · Simulacion (sin dinero real). Cada billetera arranca con su propio capital.</p>
<div class="grid">{cards}</div>
<script>
const DATA = {data};
function draw(i){{
  const d = DATA[i];
  const el = document.getElementById('c'+i);
  if(!d.points.length){{ el.parentElement.insertAdjacentHTML('beforeend','<p class="meta">Sin datos aun.</p>'); return; }}
  new Chart(el, {{
    type:'line',
    data:{{labels:d.points.map(p=>p.t), datasets:[{{
      label:'retorno acumulado %', data:d.points.map(p=>p.v),
      borderColor: i===0?'#7aa2f7':(i===1?'#4ec9b0':'#e0af68'),
      backgroundColor:'transparent', borderWidth:2,
      pointRadius: d.points.length<80?3:0, pointBackgroundColor:'#e6e6e6', tension:.15}}]}},
    options:{{responsive:true, plugins:{{legend:{{labels:{{color:'#e6e6e6'}}}},
      tooltip:{{callbacks:{{label:x=>x.parsed.y.toFixed(2)+'%'}}}}}},
      scales:{{x:{{ticks:{{color:'#9aa0a6',maxTicksLimit:8}}}},
        y:{{ticks:{{color:'#9aa0a6',callback:v=>v+'%'}}}}}}}}
  }});
}}
DATA.forEach((_,i)=>draw(i));
</script>
</body></html>"""


def main() -> None:
    exp = get_paths()["experiments"]
    series = [build_series(exp / f, lab, cap) for f, lab, cap in WALLETS]
    cards = "".join(
        f'<div class="card"><h2>{s["label"]}</h2>'
        f'<div class="meta">Capital inicial: {s["capital"]:,.0f} USD · '
        f'Estado: <span class="{"bad" if s["status"]=="QUIEBRA" else "ok"}">{s["status"]}</span></div>'
        f'<canvas id="c{i}"></canvas></div>'
        for i, s in enumerate(series))
    out = get_paths()["root"] / "docs" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out.write_text(HTML.format(gen=gen, cards=cards, data=json.dumps(series)), encoding="utf-8")
    print(f"Dashboard: {out} | " + " | ".join(f"{s['label']}={len(s['points'])}pts" for s in series))


if __name__ == "__main__":
    main()
