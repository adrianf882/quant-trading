# Informe Fase 8 — Bot de Quant Trading BTC (simulación)

Fecha: 2026-09-30 · Estado: pipeline completo, sin dinero real.

## 1. Objetivo

Construir y validar un sistema de trading **en simulación** para BTC (con opción de
otros activos), evaluado honestamente: primero contra baselines, después contra
datos nunca vistos (holdout). Objetivo real del usuario: **no perder y ganar algo**
(no necesariamente batir buy & hold).

## 2. Reglas de disciplina aplicadas

1. Cero look-ahead: feature en `t` usa solo datos hasta `t`; la señal se ejecuta en `t+1`.
2. Nada de split aleatorio: walk-forward y purged k-fold con embargo.
3. Costos siempre: comisión 0.10% + slippage 0.05% por lado.
4. Holdout final (último 20%) intocable hasta la Fase 8.
5. Cada experimento registrado en `experiments/`.
6. Baselines antes que ML.
7. Métricas de riesgo (Sharpe, Sortino, drawdown), no solo retorno.

Bibliografía usada con citas textuales: Chan, *Quantitative Trading* (2010) y
López de Prado, *Advances in Financial Machine Learning* (2018), extraídos a texto
en `reference/`.

## 3. Pipeline construido (`src/`)

`data.py` (descarga/validación), `features.py` (14 features causales),
`labels.py` (triple barrier + meta-labels), `models.py` (LightGBM + uniqueness weights),
`validation.py` (purged k-fold + walk-forward), `backtest.py` (motor con costos),
`metrics.py`, `meta.py`, `bars.py` (dollar bars), `fracdiff.py`, `pairs.py` (BTC-ETH),
`portfolio.py` (régimen), `audit.py`. 19 tests automáticos (`pytest`), todos en verde.

## 4. Resultados principales (dev, con costos)

| Estrategia | Mejor resultado neto (dev) | Sharpe | Max DD |
|---|---|---|---|
| Buy & hold BTC | +2165% total (1h) | 0.86–1.10 | -77% |
| SMA 20/50 | +47%/año a 1d | 1.16 | -39% |
| Modelo ML 3 clases | ~0% / negativo | -0.19 a 0.29 | — |
| Meta-labeling (primario tendencia) | +30%/año a 1d | 0.95 | -29% |
| Dollar bars (+fracdiff) | peor que tiempo | ≤0 | — |
| Par BTC-ETH (solo) | -34% (pierde en alcista) | -0.22 | -42% |
| **Portfolio por régimen 4h** | **+49%/año** | **1.17** | **-44%** |

Hallazgos clave:
- **Los costos dominan a alta frecuencia.** A 1h el modelo gana +162% sin costos y
  queda -100% neto (Chan, l.698).
- **El ML direccional no aporta edge** (accuracy ~43%, meta no supera a tendencia).
- **Régimen importa:** tendencia gana en alcista, el par BTC-ETH gana en no-alcista
  (anti-correlacionadas). La combinación por régimen con *hysteresis* fue lo mejor.
- Dollar bars y diferenciación fraccionaria (Prado): no mejoraron el resultado neto.

## 5. Test final — HOLDOUT (nunca visto; 2025-03-14 → 2026-09-30)

Configuración **congelada**: 4h, régimen sticky ±2%, tendencia SMA 20/50, par
BTC-ETH (lookback 84, entry 2.0, exit 0.5), fee+slip config.

| Estrategia | Ret. total | Ret. anual | Sharpe | Max DD | Ops | PF |
|---|---|---|---|---|---|---|
| **Portfolio** | +10.0% | +6.3% | **0.40** | **-26.3%** | 156 | 1.30 |
| Tendencia (solo) | +24.3% | +15.1% | 0.67 | -36.1% | 37 | 1.63 |
| Par BTC-ETH | -13.0% | -8.6% | -0.85 | -17.5% | 309 | 0.88 |
| BTC Buy & hold | +1.6% | +1.0% | 0.23 | -53.4% | 1 | — |

Retorno del Portfolio por año: 2019 +152% · 2020 +170% · 2021 +29% · **2022 -34%** ·
2023 +90% · 2024 +23% · 2025 -3% · 2026 -2%.

**Veredicto:** el Portfolio fue **positivo**, **superó a buy & hold en Sharpe
(0.40 vs 0.23)** y tuvo **mucho menor drawdown (-26% vs -53%)** en el holdout. Pero
**quedó por debajo de la simple tendencia** (Sharpe 0.67), y el par BTC-ETH perdió
también en el holdout: el "hedge" no aportó retorno, solo redujo riesgo.

## 6. Auditoría

- **Look-ahead:** OK. Test de truncado (Chan): señales idénticas con y sin datos futuros.
- **Costos:** incluidos en ambas patas de cada operación, más costos de transición entre regímenes.
- **Experimentos:** 15 archivos de evidencia, 13 familias de estrategias probadas.
- **Data-snooping:** la config final salió de 6 combinaciones (3 timeframes × 2 regímenes)
  → probable optimismo. El holdout se usó **una sola vez**, sin repetición.

## 7. Conclusión honesta

- El **producto funciona en el sentido del objetivo del usuario**: no perdió y ganó
  algo en datos nunca vistos, con menos riesgo que buy & hold.
- **Pero no es un edge contundente.** Es modesto, sensible a la selección, y el
  componente del par es un lastre que reduce riesgo pero no aporta retorno. La
  tendencia simple es más robusta y batió al portfolio en el holdout.
- **Lo más defendible hoy:** una estrategia simple de **tendencia (SMA 20/50) a 4h/1d**,
  que fue positiva, robusta entre períodos y sin parámetros frágiles. El ML no agregó valor.

## 8. Limitaciones y próximos pasos

- Muestras chicas (pocos cambios de régimen; holdout único).
- No se modeló **funding** de perpetuos (el short/par en la vida real paga/recibe carry).
- Próximos pasos posibles: (a) validar la tendencia simple en walk-forward multi-año;
  (b) mejorar el hedge ratio del par (Kalman) o descartarlo; (c) incorporar funding y
  datos intradiarios reales; (d) planificar el paper trading (Fase 9) solo si se optimiza
  el costo/rotación.

> Regla de oro que quedó demostrada: **ninguna estrategia es real hasta que sobrevive
> costos y datos nunca vistos.**

---

## 9. Actualización — tests posteriores y Fase 9

Tras la Fase 8 se probaron, sin dejar pendientes, los métodos restantes de Chan/Prado:

| Test | Resultado | Veredicto |
|---|---|---|
| Sizing vol-target (Kelly-like) | Sharpe 0,70 → 0,56 (4h) | ✗ empeora |
| Hedge por **Kalman** en el par | Par 4h: -42,5% → **+79,8%** (Sharpe -0,28 → 0,54) | ✓ mejora |
| Walk-forward con re-optimización móvil | 4h: 0,70 → **1,03**; 1d: empeora | ✓ solo a 4h |
| **Funding** (carry) | Tendencia long: 379% → 205% (Sharpe 0,70 → 0,56); par ~neutro | informativo / costo real |
| Allocator por **autocorrelación** (strategy selection) | Todas las configs con Sharpe negativo | ✗ falla (estadístico ruidoso) |

**Hallazgo:** el **regime switching** funciona solo con el régimen **lento/pegajoso** (SMA200 + hysteresis); con un estadístico rápido (autocorrelación) se whippea y pierde.

### Candidato final: Portfolio B-Kalman (4h)
Régimen sticky ±2% → tendencia SMA 20/50 en alcista, par BTC-ETH con hedge de
**Kalman** en no-alcista.

| | Ret. anual | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| **B-Kalman (todo)** | +45,3% | **1,18** | 1,75 | **-38,9%** |
| B-Kalman (holdout) | +10,0% | 0,54 | 0,80 | -16,3% |
| BTC Buy & hold | +49,6% | 0,97 | 1,39 | -77,0% |

Le gana a buy & hold en **Sharpe y drawdown**, con retorno algo menor.

### Fase 9 — Paper trading
`src/paper.py`: descarga las últimas velas, reconstruye B-Kalman con la config
congelada y registra posición objetivo + P&L simulado en `experiments/paper_log.csv`.
Pensado para correr cada 4h (Programador de tareas). **Sin dinero real.**

Caveat final: la config salió de muchas pruebas; el holdout ya se usó. Los
resultados son el **mejor caso plausible**, a confirmar en paper trading real.

## 10. Revisión a la luz de Jansen (*ML for Algorithmic Trading*, 2ª ed.)

| Test (Jansen) | Resultado | Veredicto |
|---|---|---|
| **IC de features** (Spearman vs retorno siguiente) | \|IC\| ≤ 0,055 (1d), ≤ 0,08 (4h); ninguna feature con capacidad predictiva | features sin señal |
| **Ensemble** (RF + GBM + LightGBM) vs LightGBM solo | accuracy 0,37 vs 0,36 (1d); 0,42 vs 0,42 (4h); Sharpe sin mejora | ensemble no ayuda |

**Conclusión:** el pipeline de Jansen (ensembles, IC) **confirma** lo ya visto: **no
hay edge en el ML direccional sobre precio**. Las herramientas no cambian el
resultado; el hallazgo es robusto a los tres libros (Chan, Prado, Jansen).

## 11. Infraestructura

- Repositorio privado: https://github.com/adrianf882/quant-trading
- **Paper trading desatendido** vía GitHub Actions (`.github/workflows/paper.yml`):
  corre `src/paper.py` cada 4h y commitea `experiments/paper_log.csv`. Sin API keys
  (solo datos públicos) → riesgo financiero nulo en esta etapa.
