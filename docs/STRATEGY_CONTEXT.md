# btc-quant — Contexto estratégico para nuevo chat

> Pega este documento completo al inicio de un nuevo chat de Claude para continuar
> el desarrollo de la estrategia de inversión. Cubre decisiones tomadas, diseño actual,
> resultados validados, rationale de cada parámetro y próximos pasos.

---

## 1. Objetivo del proyecto

Sistema cuantitativo de inversión en BTC/USDT con dos capas:

- **Capa 1 (STRAT-06) — IMPLEMENTADA Y VALIDADA:** DCA modulated + deep-value reserve.
  Acumulación sistemática de BTC con sesgo contrarian. Sin ventas. Largo plazo.
- **Capa 2 (STRAT-07) — PENDIENTE DE DISEÑO:** Overlay táctico multi-señal sobre el
  BUFFER pool. Entradas adicionales basadas en señales técnicas (tampoco vende en v1).

**Prioridad absoluta:** Preservación de capital primero, maximización de retorno segundo.
Nada va a producción sin backtest completo + walk-forward + auditoría.

---

## 2. Filosofía de inversión acordada

- Horizonte temporal: 5–10 años.
- Asset: BTC/USDT únicamente en esta fase.
- Inflow mensual: €500/mes (configurable).
- Exchange: Binance.
- No apalancamiento. No shorts. No ventas en esta fase.
- Fees: 0.1% por operación (Binance estándar).

---

## 3. STRAT-06 — Diseño completo

### 3.1 Arquitectura de pools

| Pool | % del inflow | EUR/mes | Función |
|---|---|---|---|
| BASELINE | 55% | €275 | DCA semanal modulated; recibe refuerzo de BUFFER en dips |
| BUFFER | 20% | €100 | Dry powder — cubre el exceso del multiplier; recibe overflow de RESERVE |
| RESERVE | 25% | €125 | Deep-value reserve — solo se despliega en crashes severos (≥−50%) |

**Inflow mensual €500:** BASELINE €275 + BUFFER €100 + RESERVE €125.

#### Rationale del split 55 / 20 / 25

- **55% BASELINE:** El DCA semanal es el motor principal de acumulación. Debe ser la
  fracción dominante. Con 55%, la compra base semanal es €68.75 (≈ el compromiso mínimo
  por semana independientemente del mercado).
- **20% BUFFER:** Con un multiplier máximo de 2.5× y una compra base de €68.75, el
  máximo semanal es €171.88. La diferencia (€103.13) puede provenir del BUFFER. Con 20%
  del inflow (€100/mes) el BUFFER se recarga suficientemente rápido para cubrir semanas
  de alta modulación sin vaciarse.
- **25% RESERVE:** Necesita acumular €1,500 (el cap) en 12 meses. 12 × €125 = €1,500.
  El 25% fue elegido para que el RESERVE esté lleno y listo exactamente al cabo de un año
  de inflows. Si el bear llega antes del año 1, el RESERVE tendrá lo que haya acumulado
  hasta ese punto (comportamiento correcto — parcialmente funding es mejor que nada).

---

### 3.2 Mecánica BASELINE (compra semanal modulated)

- Compra **cada lunes** (weekday 0, configurable).
- Tamaño base (unmodulated): `baseline_per_week_eur = monthly_inflow × baseline_pct / 4`
  = €500 × 0.55 / 4 = **€68.75/semana**.
- **Multiplier por drawdown desde ATH** (implementado en `DrawdownMultiplier`):

| Drawdown desde ATH | Multiplier |
|---|---|
| 0% a −10% | **1.0×** (sin amplificación) |
| −10% a −20% | **1.25×** |
| −20% a −35% | **1.50×** |
| −35% a −50% | **2.00×** |
| ≥−50% | **2.50×** |

- Gasto semanal efectivo = `baseline_per_week_eur × multiplier`. Si BASELINE no tiene
  suficiente cash, **BUFFER cubre el déficit** (transfer BUFFER → BASELINE en el mismo bar).
- `max_concentration_pct = 1.0` (guard desactivado por diseño — ver §7).
- `min_order_eur = 10.0` (no se ejecuta si el importe es menor).

#### Rationale de los multipliers

Los valores son deliberadamente conservadores (máximo 2.5×, no 4× o 10×):
- Un multiplier excesivo vaciaría el BUFFER en pocas semanas de bear market, eliminando
  la capacidad de modular durante el resto del crash.
- Los tramos están separados con más anchura en la zona media (−20% a −35% es un tramo
  de 15pp) porque los drawdowns entre −20% y −35% son frecuentes en BTC y no merecen
  la misma reacción que un crash de −50%.
- La escala 1.0 → 1.25 → 1.5 → 2.0 → 2.5 es aproximadamente geométrica (ratio ~1.25
  por tramo), lo que da una curva de respuesta progresiva sin saltos bruscos.

---

### 3.3 Mecánica RESERVE (deep-value reserve)

- **Cap:** `reserve_cap_eur = reserve_cap_months × monthly_inflow × reserve_pct`
  = 12 × €500 × 0.25 = **€1,500**.
- El exceso sobre el cap se redirige automáticamente a BUFFER (no se pierde).
- Se activa cuando el drawdown desde ATH ha estado **≥ −50% durante ≥ 7 días consecutivos**
  (anti-flash-crash — filtra recuperaciones rápidas como marzo 2020).
- **4 tranches** que se despliegan en cascada (porcentaje del balance actual, no del cap):

| Tranche | Drawdown trigger | % del balance de RESERVE |
|---|---|---|
| 1 | ≥ −50% sostenido 7d | 20% |
| 2 | ≥ −60% sostenido 7d | 25% |
| 3 | ≥ −70% sostenido 7d | 25% |
| 4 | ≥ −75% sostenido 7d | 30% |

- Cada tranche se marca como "usado" tras disparar y no vuelve a activarse hasta un reset.
- **Reset de ciclo:** cuando el precio alcanza un nuevo ATH, los 4 tranches se resetean
  y el RESERVE queda listo para el siguiente bear cycle.
- En un bear cycle completo, el total deployable es el 80% del balance
  (20+25+25+30 = 100%, pero el 30% del tranche 4 se aplica sobre el ~45% que queda
  tras los primeros 3, resultando en ~57.5% del balance original; el 20% restante queda
  como reserva permanente).

#### Rationale del diseño de tranches

- **Triggers −50 / −60 / −70 / −75%:** En los dos grandes bear markets de BTC
  (2018: −83% máx, 2022: −74% máx), drawdowns de −50% a −75% ocurrieron en ambos.
  No se diseñó para −80% porque la cantidad que queda en RESERVE a ese nivel es pequeña
  (ya se han disparado 3 tranches) y añadir un quinto tranche aumentaría la complejidad
  sin beneficio material.
- **Persistencia 7 días:** Flash crashes (horas/días) no merecen desplegar capital de
  emergencia. 7 días garantiza que es un crash sostenido. En 2020 (crash de marzo,
  −50% en 2 días y recuperación en semanas), el tranche NO habría disparado.
- **% progresivos (20/25/25/30):** Desplegamos más capital cuanto más profundo es el
  crash porque la convicción del "comprar en mínimos" aumenta. El primer tranche es
  pequeño (20%) porque −50% en BTC no es excepcional; el cuarto (30%) es el mayor
  porque −75% sostenido 7 días es un evento raro y el momento más favorable.
- **Reserve_cap_months = 12:** El RESERVE debe estar lleno antes de que llegue el primer
  bear market. Históricamente BTC tiene ciclos bull de 1-3 años. 12 meses de acumulación
  (€1,500) es suficiente para tener "munición" lista. El exceso después del año 1 va a
  BUFFER como dry powder adicional.

---

### 3.4 Flujo de transfers en cada bar

En orden dentro de `generate_signals()`:
1. **RESERVE → BUFFER** (overflow si balance RESERVE > cap)
2. **RESERVE → BASELINE** (deploy tranche si trigger activo y persistencia cumplida)
3. **BUFFER → BASELINE** (cubrir déficit si baseline × multiplier > cash disponible)

Estos transfers ocurren ANTES de emitir la señal de compra. La señal lleva el importe
total que BASELINE tiene disponible (inflows + transfers recibidos ese bar).

---

## 4. Resultados validados (backtest histórico)

**Período:** 2017-08-17 → 2026-05-06 (8.73 años, 3,185 barras diarias)
**Capital total inyectado:** €52,500 (105 meses × €500)
**Execution:** 0.1% fee, slippage regime-aware (`ExecutionSimulatorV2`)

| Estrategia | Valor final | Retorno | Max Drawdown | Sharpe | Sortino | Idle cash |
|---|---|---|---|---|---|---|
| Buy-and-Hold | €986,072 | +1,778% | −83.2% | 0.84 | 1.13 | €53 |
| DCA Benchmark | €285,998 | +445% | −74.8% | 1.32 | 2.07 | €424 |
| **STRAT-06** | **€273,105** | **+420%** | **−73.9%** | **1.34** | **2.17** | **€5,105** |

**Conclusión validada:** STRAT-06 ≈ DCA en retorno absoluto (−4.5%), pero mejor
Sharpe (+0.02), Sortino (+0.10) y MaxDD (+0.9pp más shallow). Los €5,105 idle son
intencionales: €1,500 RESERVE (permanente, dry powder para próximo bear) + €3,468 BUFFER
(overflow de 8.5 años de inflows sin usar) + €138 BASELINE (residual de último buy).

**Reserve deployments históricos (8 tranches en 2 bear cycles):**

| Fecha | Drawdown | Desplegado | Precio BTC |
|---|---|---|---|
| 2018-02-07 | −60.2% | €150 | €7,599 |
| 2018-04-04 | −64.4% | €213 | €6,796 |
| 2018-11-22 | −77.1% | €375 | €4,370 |
| 2018-11-26 | −79.8% | €338 | €3,862 |
| 2022-05-15 | −53.6% | €300 | €31,329 |
| 2022-06-18 | −71.9% | €331 | €18,971 |
| 2022-07-05 | −70.1% | €280 | €20,176 |
| 2022-11-22 | −76.0% | €402 | €16,227 |

Total deployed por RESERVE: €2,388 a precios €3,862–€31,329/BTC.

**Baseline execution:** 398/455 Mondays ejecutados (87.5%). Multiplier medio: 1.81×.
57 bloqueados por cash flow timing (esperando el próximo inflow mensual — normal).
0 bloqueados por concentration guard (guard desactivado).

---

## 5. Stack técnico

```
Lenguaje:       Python 3.11+
Deps:           uv (gestión), pytest, pandas, numpy, matplotlib
Layout:         src/btc_quant/ (src layout)
Exchange:       Binance (no live todavía, LIVE_TRADING_ENABLED=false)
Data:           data/processed/btcusdt_1d.parquet (2017-2026, OHLCV diario)
```

**Módulos implementados:**
```
src/btc_quant/
├── backtester/
│   ├── engine.py               # BacktestEngine principal
│   ├── execution_simulator.py  # ExecutionSimulatorV2 (regime-aware slippage)
│   ├── inflow_scheduler.py     # InflowScheduler.monthly()
│   ├── metrics/
│   │   ├── calculator.py       # MetricsCalculator (Sharpe, Sortino, MaxDD, etc.)
│   │   └── report.py           # MetricsReport, StrategyMetrics dataclasses
│   ├── models/                 # Order, Pool, Portfolio, Position, Signal, Trade, etc.
│   ├── regime_detector.py      # RegimeDetector para slippage calibration
│   └── strategy_base.py        # StrategyBase ABC
├── data/                       # Downloader, cleaner, validator, storage
├── indicators/                 # Trend, oscillators, volatility, volume, structure
├── strategies/
│   ├── buy_and_hold.py         # BuyAndHoldStrategy (benchmark)
│   ├── dca_benchmark.py        # DCABenchmarkStrategy (benchmark)
│   └── strat06/
│       ├── ath_tracker.py      # ATHTracker (ATH + drawdown tracking)
│       ├── config.py           # DCAModulatedConfig (todos los parámetros)
│       ├── drawdown_multiplier.py  # DrawdownMultiplier (tabla de multipliers)
│       ├── reserve_manager.py  # ReserveManager (4 tranches + cycle reset)
│       └── strategy.py         # DCAModulatedStrategy (orchestrator)
```

---

## 6. Artefactos de backtest

```
data/reports/strat06_comparative_20260511_193601/
├── equity_curve_buy_and_hold.csv
├── equity_curve_dca_benchmark.csv
├── equity_curve_strat06.csv
└── comparative_summary.json        ← fuente canónica de métricas

notebooks/
├── 03_strat06_comparative_analysis.ipynb  (análisis cuantitativo)
└── 04_strat06_visual_comparison.ipynb     (visualizaciones comparativas, 7 figuras)

docs/reports/
├── STRAT_06_DESIGN_REFINEMENT_001.md      (decisión de eliminar concentration guard)
└── STRAT_06_HISTORICAL_VALIDATION.md      (reporte de validación completo)

scripts/
├── run_strat06_comparative_backtest.py    (reproduce el backtest)
└── diagnose_strat06_deployment.py         (diagnóstico de deployment por bar)
```

---

## 7. Decisiones de diseño tomadas — no revertir

### 7.1 Concentration guard desactivado (max_concentration_pct = 1.0)

**Problema:** Con el guard a 0.70, BTC superó el 70% de concentración de portfolio en
noviembre 2017 (3 meses después del inicio) y nunca volvió a bajar. Resultado: 79.8%
de los Mondays bloqueados, 69% del capital idle.

**Decisión:** `max_concentration_pct = 1.0` (efectivamente desactivado). El campo
existe en el código para variantes futuras que lo necesiten.

**Por qué NO es overfitting:**
1. El cambio no optimiza ninguna métrica sobre datos históricos. Corrige un comportamiento
   patológico donde la estrategia no invertía su capital declarado.
2. No se toca ningún threshold numérico observado en el backtest.
3. La decisión se tomó y documentó ANTES de ver las métricas comparativas vs benchmarks.
4. El rationale es estructural: la concentración en BTC es el OBJETIVO de STRAT-06,
   no un riesgo a limitar. El guard era correcto en un contexto de portfolio diversificado
   (SYSTEM_FINAL §4) pero patológico para acumulación pura.

**Alternativas descartadas:**
- **Opción B** — Recalcular concentración sobre capital invertido acumulado
  (BTC_value / capital_invertido): cambia el significado semántico del campo y confunde
  a futuros mantenedores. La concentración de mercado es más intuitiva.
- **Opción C** — Bypass del guard solo para compras pequeñas de baseline: añade
  complejidad sin atacar la causa raíz. El comportamiento patológico persistiría
  para compras moduladas grandes.

**Referencia:** `docs/reports/STRAT_06_DESIGN_REFINEMENT_001.md`

---

### 7.2 calculate_returns salta ceros iniciales

**Problema:** Las curvas de equity de DCA/STRAT-06 empiezan en €0 (sin capital inicial,
solo inflows mensuales). El cálculo original `diff(values) / values[:-1]` producía NaN
en el primer bar (0/0), propagando NaN a Sharpe y Sortino.

**Decisión:** Calcular returns desde el **primer bar no-cero** de la curva, ignorando
el prefijo de ceros.

**Alternativas descartadas:**
- Rellenar €0 con €1 como valor ficticio: distorsiona el retorno del primer período.
- Usar `.fillna(0)` en returns: convierte NaN en 0%, lo cual es incorrecto (no hubo
  retorno negativo, simplemente no había capital aún).

---

## 8. Próximo paso — STRAT-07 (pendiente de diseño)

STRAT-07 es el overlay táctico que añade entradas adicionales sobre el **BUFFER pool**
de STRAT-06. No está diseñado todavía.

**Contexto:** Tras 8.5 años de backtest, el BUFFER acumuló €19,737 pero solo desplegó
€1,252 (6.3%), porque el deploy de BUFFER depende de que BASELINE esté comprando
(y el multiplier consuma más de lo que BASELINE tiene). STRAT-07 daría al BUFFER una
fuente de señales propias, independiente del timing de BASELINE.

**Indicadores disponibles** (implementados en `src/btc_quant/indicators/`):
- **Trend:** EMA, SMA, DEMA, TEMA, WMA, HullMA, KAMA, SuperTrend, VWAP
- **Oscillators:** RSI, Stochastic RSI, MACD, Williams %R, CCI, Ultimate Oscillator, MFI
- **Volatility:** Bollinger Bands, ATR, Keltner Channel, Donchian Channel, Historical Vol
- **Volume:** OBV, VWAP, CMF, ADLine, Force Index
- **Structure:** Pivot Points, Support/Resistance, Fibonacci, Market Structure

**Restricciones de diseño para STRAT-07:**
1. Solo BUY — no SELL en esta fase.
2. Solo usa capital del BUFFER pool.
3. No puede vaciar el BUFFER por completo (necesita reserva para cubrir modulación de
   BASELINE en bears).
4. Requiere backtest propio + integración con STRAT-06 antes de producción.

---

## 9. Limitaciones conocidas del backtest actual

1. **Single asset, single period.** No hay walk-forward. Los parámetros no han sido
   optimizados (se diseñaron a priori) pero tampoco han sido validados fuera de muestra.
2. **IRR no implementado.** Para estrategias con inflows periódicos, el Sharpe/Sortino
   se calculan sobre la curva de equity pero el retorno total se mide sobre el capital
   inyectado (no CAGR, que asume lump-sum).
3. **Todos los Mondays al cierre de barra.** No hay variación intra-semana ni intra-día.
4. **Reserve cap puede ser demasiado pequeño.** En 8.5 años de bull con solo 2 bear
   markets, €9,237 overflowed de RESERVE a BUFFER. Con cap de €1,500, el RESERVE
   se llena en 12 meses y todo lo que sigue va a BUFFER. Puede valer la pena subir
   `reserve_cap_months` si el próximo ciclo bull dura 3+ años.
5. **STRAT-07 no activo.** BUFFER deploying only via multiplier modulation, no via
   tactical signals.

---

## 10. Reglas de sesión para Claude Code (este repo)

- Todo código en inglés. Documentos pueden ser en español.
- No escribir código de ejecución live sin instrucción explícita.
- No hacer commits sin que Roger lo pida.
- No añadir features no pedidas ni over-engineer.
- Cada función crítica necesita test unitario.
- `LIVE_TRADING_ENABLED=false` es el safety gate — nunca tocar sin instrucción explícita.
- No hardcodear credenciales.
