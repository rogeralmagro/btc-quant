# TradingView Pine Scripts — F5.1 Cross-Validation

## Configuración del chart

| Parámetro | Valor |
|---|---|
| Símbolo | `BINANCE:BTCUSDT` |
| Timeframe | `1D` |
| Rango | `2017-08-17 → 2026-05-06` |
| Escala | Logarítmica (recomendada) |

## Scripts

| Archivo | Indicador | overlay |
|---|---|---|
| `strat06_bah.pine` | BAH Benchmark (F5.1) | `false` (pane separado) |
| `strat06_dca_benchmark.pine` | DCA Benchmark (F5.1) | `false` (pane separado) |
| `strat06_main.pine` | STRAT-06 (F5.1) | `true` (sobre precio) |

## Cómo cargar en TradingView

1. Abre Pine Script Editor (botón inferior del chart)
2. Pega el contenido del archivo `.pine`
3. Haz clic en **Add to chart**
4. Repite para los otros 2 scripts

## Verificaciones F5.1

### Verificación 1 — Deployments de RESERVE (binaria: pasa/no pasa)

`strat06_main.pine` muestra triángulos rojos (`▲`) debajo de las velas en las fechas
de deployment. Deben aparecer exactamente en:

| Fecha | Drawdown | Despliegue |
|---|---|---|
| 2018-02-07 | ≈ −60% | ≈ €150 |
| 2018-04-04 | ≈ −64% | ≈ €213 |
| 2018-11-22 | ≈ −77% | ≈ €375 |
| 2018-11-26 | ≈ −76% | ≈ €338 |
| 2022-05-15 | ≈ −54% | ≈ €300 |
| 2022-06-18 | ≈ −72% | ≈ €331 |
| 2022-07-05 | ≈ −70% | ≈ €280 |
| 2022-11-22 | ≈ −76% | ≈ €402 |

### Verificación 2 — Cost basis final (tolerancia ±5%)

| Estrategia | Objetivo | Rango aceptable |
|---|---|---|
| STRAT-06 | €8,775/BTC | €8,336 – €9,214 |
| DCA Benchmark | €14,973/BTC | €14,224 – €15,722 |

Ver dashboard (tabla top-right) al final del rango.

### Verificación 3 — BTC acumulado final (tolerancia ±5%)

| Estrategia | Objetivo | Rango aceptable |
|---|---|---|
| STRAT-06 | 3.29 BTC | 3.13 – 3.45 |
| DCA Benchmark | 3.51 BTC | 3.33 – 3.69 |

## Notas

- Los scripts usan precios USDT tratando USDT ≈ EUR (el backtest usa BTCUSDT data).
  Las cifras de BTC acumulado son idénticas en cualquier divisa. El cost basis
  puede diferir por tipo de cambio EUR/USD histórico.
- Si alguna verificación falla: STOP. No cerrar F4. Investigar divergencia.
