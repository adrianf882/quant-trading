# trading

Proyecto de Quant Trading (fase de **simulación**, sin dinero real).

Objetivo: descargar datos de BTC, construir features sin lookahead, etiquetar con
triple barrier, entrenar un modelo de ML, validarlo con métodos aptos para series
temporales, backtestear con costos realistas y correrlo en paper trading.

> Estado actual: **Fase 0 (setup)**.

## Requisitos

- Python 3.10+ (este entorno usa 3.10.11)
- Windows + PowerShell

## Puesta en marcha

```powershell
# 1. Crear el entorno virtual
py -3.10 -m venv .venv

# 2. Activarlo
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

# 4. Verificar
py -c "import pandas, ccxt, lightgbm; print('ok')"
```

## Estructura

```
trading/
├── config.yaml         # parámetros del proyecto
├── requirements.txt
├── data/
│   ├── raw/            # OHLCV descargado (parquet)
│   └── processed/      # features + labels
├── src/                # código del proyecto
├── notebooks/          # exploración
├── experiments/        # log de experimentos
├── reference/          # material de estudio (texto de libros)
├── tools/              # utilidades (ej: extraer epub)
└── tests/              # tests automáticos
```

## Fases

- [x] 0. Setup
- [ ] 1. Datos
- [ ] 2. Baselines
- [ ] 3. Features
- [ ] 4. Etiquetado (triple barrier)
- [ ] 5. Modelo baseline (LightGBM)
- [ ] 6. Validación temporal
- [ ] 7. Backtest realista
- [ ] 8. Test final y auditoría
- [ ] 9. Paper trading

## Reglas innegociables

1. Cero lookahead: feature en `t` usa solo datos hasta `t`; la señal se ejecuta en `t+1`.
2. Nada de `train_test_split` aleatorio: solo split temporal / walk-forward / purged k-fold.
3. Costos siempre incluidos (comisión 0.1% por lado, slippage 0.05%).
4. Set de test final intocable (último ~20%), usado una sola vez.
5. Registrar cada experimento.
6. Baselines antes que ML.
7. Medir riesgo (Sharpe, Sortino, drawdown), no solo retorno.
