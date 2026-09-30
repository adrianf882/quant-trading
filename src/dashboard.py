"""Genera un dashboard HTML autocontenido (2 graficos de retorno acumulado).

Lee experiments/paper_log_futures.csv y experiments/paper_log_spot.csv y escribe
docs/index.html con Chart.js (CDN publico). Listo para GitHub Pages.

Uso:
    py -m src.dashboard
"""
import json
from pathlib import Path

import pandas as pd

from .config import get_paths


def build_series(csv_path: Path, label: str) -> dict:
    if not csv_path.exists():
        return {"label": label, "points": []}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"label": label, "points": []}
    base = float(df["equity"].iloc[0])
    cum = df["equity"] / base - 1
    pts = [{"t": str(df["bar"].iloc[i]), "v": round(float(cum.iloc[i]) * 100, 3)}
           for i in range(len(df))]
    return {"label": label, "points": pts}


HTML = """<!doctype html>
<html lang="es"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quant Trading - Paper Trading</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body{{font-family:system-ui,Arial,sans-serif;background:#0f1117;color:#e6e6e6;margin:0;padding:24px}}
  h1{{font-size:20px;font-weight:600}} .sub{{color:#9aa0a6;font-size:13px;margin-top:-8px}}
  .grid{{display:grid;grid-template-columns:1fr;gap:20px;max-width:900px;margin:auto}}
  .card{{background:#171a21;border:1px solid #262a33;border-radius:12px;padding:16px}}
  .card h2{{font-size:15px;margin:0 0 10px}} canvas{{max-height:320px}}
</style></head>
<body>
<h1>Paper Trading - retorno acumulado</h1>
<p class="sub">Generado: {gen} · Simulacion (sin dinero real) · fuente: logs del cron</p>
<div class="grid">
  <div class="card"><h2>{lab0}</h2><canvas id="c0"></canvas></div>
  <div class="card"><h2>{lab1}</h2><canvas id="c1"></canvas></div>
</div>
<script>
const DATA = {data};
function draw(i){{
  const d = DATA[i]; if(!d.points.length) return;
  new Chart(document.getElementById('c'+i), {{
    type:'line',
    data:{{labels:d.points.map(p=>p.t), datasets:[{{label:'retorno acumulado %',
      data:d.points.map(p=>p.v), borderColor:i?'#4ec9b0':'#7aa2f7',
      backgroundColor:'transparent', borderWidth:2, pointRadius:0, tension:.15}}]}},
    options:{{responsive:true, plugins:{{legend:{{labels:{{color:'#e6e6e6'}}}},
      tooltip:{{callbacks:{{label:x=>x.parsed.y.toFixed(2)+'%'}}}}}},
      scales:{{x:{{ticks:{{color:'#9aa0a6',maxTicksLimit:8}}}},
        y:{{ticks:{{color:'#9aa0a6',callback:v=>v+'%'}}}}}}}}
  }});
}}
draw(0); draw(1);
</script>
</body></html>"""


def main() -> None:
    paths = get_paths()
    exp = paths["experiments"]
    futures = build_series(exp / "paper_log_futures.csv", "B-Kalman (futuros 4h)")
    spot = build_series(exp / "paper_log_spot.csv", "SPOT trend (1d)")
    out = paths["root"] / "docs" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out.write_text(
        HTML.format(gen=gen, lab0=futures["label"], lab1=spot["label"],
                    data=json.dumps([futures, spot])),
        encoding="utf-8")
    print(f"Dashboard: {out} | puntos futures={len(futures['points'])} "
          f"spot={len(spot['points'])}")


if __name__ == "__main__":
    main()
