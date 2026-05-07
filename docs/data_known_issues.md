# Known Data Quality Issues

## BTCUSDT 4h — Historical Gaps (2018)

### Descripción del problema

Los datos de BTCUSDT 4h descargados de Binance contienen **8 gaps temporales**,
todos concentrados en 2018. Son huecos reales debidos a cortes técnicos de Binance
durante sus primeros meses de operación; no son errores de descarga ni de paginación.

Verificado: los mismos gaps aparecen en fuentes de datos independientes
(CryptoDataDownload, Kaiko) para el mismo período.

### Gaps detectados

| # | Inicio gap | Fin gap | Velas faltantes | Duración |
|---|------------|---------|-----------------|----------|
| 1 | 2018-02-08 | 2018-02-09 | 7 | 28h |
| 2 | 2018-06-26 | 2018-06-26 | 2 | 8h |
| 3 | 2018-07-04 | 2018-07-04 | 1 | 4h |
| 4–8 | 2018 | 2018 | varios | < 48h |

El gap más significativo (7 velas = 28h) coincide con el conocido apagón de
Binance del 8-9 de febrero de 2018, cuando la plataforma suspendió operaciones
por mantenimiento de emergencia durante el primer crash de BTC de ese año.

### Política aplicada: Opción E (forward-fill con marcador)

**Principio**: todas las velas ausentes cuya duración sea ≤ 48h se rellenan con
velas sintéticas. Las velas sintéticas se marcan explícitamente con `is_synthetic=True`.

**Valores de velas sintéticas** (zero look-ahead bias):
```
open = high = low = close = close de la última vela real anterior al gap
volume = 0
quote_volume = 0
num_trades = 0
is_synthetic = True
```

El precio de relleno es el `close` de la última vela real **antes** del gap.
Nunca se usa el `open` de la vela posterior (eso sería look-ahead).

**Umbral de rechazo**: gaps > 48h no se rellenan. En caso de existir, el
`DataValidator` los detecta como gaps reales y marca el dataset como inválido.
Los 8 gaps de 2018 son todos < 48h, por lo que todos se rellenan.

### Implicaciones para backtesting

1. **Indicadores**: se calculan sobre TODAS las velas (reales y sintéticas).
   En zonas sintéticas el precio no varía y el volumen es cero, por lo que
   los indicadores de momentum (RSI, MACD) reflejarán ausencia de movimiento.

2. **Estrategia**: antes de generar cualquier señal, la capa de estrategia
   llama a `is_period_clean(df, end_idx, lookback)` (ver `data/quality.py`).
   Si la ventana de lookback contiene alguna vela sintética, la estrategia
   no genera señal y no abre posición.

3. **Reporting**: el `ValidationReport` incluye `has_synthetic_candles=True` y
   `synthetic_count=N` cuando hay velas sintéticas. El script de descarga
   muestra el conteo en la tabla resumen y marca el status como `OK*`.

### Módulos relevantes

| Módulo | Responsabilidad |
|--------|-----------------|
| `data/cleaner.py` | `GapFiller.fill_gaps()` — detección y relleno |
| `data/validator.py` | `DataValidator.validate()` — reporting de sintéticas |
| `data/quality.py` | `is_period_clean()` — exclusión en estrategia |
| `data/downloader.py` | `OHLCV_COLUMNS` incluye `is_synthetic` |

### Decisión de diseño: ¿Por qué forward-fill en lugar de interpolación?

La interpolación lineal entre `prev_close` y `next_open` introduce **look-ahead
bias**: en el momento del gap, no tenemos información del futuro. El forward-fill
con `prev_close` usa únicamente información disponible en ese instante y es el
estándar de la industria para gaps en datos OHLCV.
