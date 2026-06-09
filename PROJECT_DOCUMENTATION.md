# Informe del proyecto: Pronóstico horario de grCO2 (completo)

**Última actualización:** 2026-06-08

**Resumen ejecutivo**
- Objetivo: predecir la variable `grco2_kwh` para los próximos 48 horas mediante un modelo por horizonte (XGBoost) y cuantificar incertidumbre con bootstrap sobre residuales OOS.
- Resultado destacado (h=1): Modelo MAE = 4.8145, RMSE = 6.4001; Baseline MAE = 6.9026, RMSE = 8.6993 (mejora ≈ 30.3%).

**Índice**
- Contexto y datos
- Entorno
- Metodología
  - Ingeniería de features
  - Creación de targets
  - Validación temporal y búsqueda de hiperparámetros
  - Entrenamiento por horizonte
  - Intervalos de confianza por bootstrap
  - Baseline y métricas
- Resultados
  - Métricas por horizonte (extracto)
  - Muestra de pronóstico 48h
  - Observaciones sobre rendimiento por horizonte
  - Espacio para imágenes (placeholders)
- Artefactos generados
- Reproducción paso a paso
- Uso de modelos y cómo obtener resultados
  - Exportar resultados sin re-entrenar
  - Serialización de modelos y pronósticos desde modelos guardados
- Solución de problemas comunes
- Recomendaciones y siguientes pasos

---

**Contexto y datos**
- Fuente de datos: archivo parquet (ej. `part-00000-4f2b0623-696a-42ac-a276-7ba8ad7618ad-c000.snappy.parquet`) que contiene series horarias y varias columnas de lags/estadísticos ya preprocesados.
- Columnas principales detectadas: `dateh`, `dateh_ts`, `grco2_kwh`, lags `grco2_kwh_lag_1h`..`grco2_kwh_lag_24h`, medias/aggregados (`grco2_kwh_avg_1d`..), flags calendario (`year`, `month`, `day`, `dayofweek`, `es_festivo`, `es_fin_semana`, etc.), codificaciones cíclicas (month_sin/cos, dayofweek_sin/cos, ...), y aportes/exógenas (`total_aportes_lag1`, `total_aportes_avg7d`, ...).
- Periodo usado en el experimento de referencia:
  - Entrenamiento: 2022-02-07 00:00:00 -> 2025-11-29 23:00:00 (33,480 filas)
  - Test: 2025-11-30 00:00:00 -> 2025-12-29 23:00:00 (720 filas)

**Entorno**
- Sistema: macOS (Homebrew disponible). Nota: XGBoost nativo en macOS suele requerir `libomp`.
- Entorno Python: ejecutado en el intérprete `pyenv` `.envMaestria`.
- Librerías principales: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib` (recomendado para serializar).

---

## Metodología

### Preparación de datos
- Lectura del parquet y conversión de `dateh` a `datetime`, orden y `set_index('dateh')`.
- Target principal: `grco2_kwh`.

### Ingeniería de features
- Se reutilizan las features ya presentes en el parquet (lista incluida en el repositorio). Evitamos recrear duplicados.
- Nuevas features generadas por `crear_features()`:
  - `hour`, `hour_sin`, `hour_cos` (codificación cíclica de la hora del día).
  - `is_weekend` (flag binario si dayofweek ∈ {5,6}).
  - Lags largos: `lag_48`, `lag_72`, `lag_168`.
  - Estadísticos móviles sobre `y_past = target.shift(1)`: `roll_std_W`, `roll_min_W`, `roll_max_W` para ventanas W ∈ {24,48,72,168}.
- Antes de pasar los features a XGBoost se filtran: `X = X.select_dtypes(include=[np.number, "bool"])` para evitar errores con tipos datetime u object.

### Creación de targets
- Se crean columnas `target_h1` .. `target_h48` con `df[target].shift(-h)` para cada horizonte h.
- Se eliminan filas con NaNs resultantes antes de proceder.

### Validación temporal y búsqueda de hiperparámetros
- Validación: `TimeSeriesSplit(n_splits=5, test_size=7*24)` (cada fold valida 7 días) para obtener particiones fuera de muestra (OOS) realistas.
- Búsqueda de hiperparámetros: `RandomizedSearchCV` con `n_iter=30`, `scoring='neg_mean_absolute_error'`, `cv=tscv`, paralelo `n_jobs=-1`.
- Horizontes clave optimizados: 1, 6, 12, 24, 48. Los mejores parámetros resultantes (`best_params_by_horizon`) se asignan a otros horizontes por reglas simples (función `seleccionar_params(h)`):
  - h ≤ 3 → params de h=1
  - 4 ≤ h ≤ 9 → params de h=6
  - 10 ≤ h ≤ 18 → params de h=12
  - 19 ≤ h ≤ 36 → params de h=24
  - h ≥ 37 → params de h=48

### Entrenamiento final por horizonte
- Para cada horizonte h = 1..48 se instancia `XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **params)` y se entrena con `model.fit(X_train, Y_train[f"target_h{h}"])`.
- Cada modelo se guarda en un diccionario `modelos[h]` durante la ejecución.

### Residuales fuera de muestra y bootstrap para IC
- Para cada horizonte h se re-entrena por fold (usando `seleccionar_params(h)`) y se predice sobre la validación del fold.
- Se calculan residuales OOS por fold: `y_val - y_pred` y se concatenan en `residuales_por_horizonte[h]` (en el experimento hay ≈840 residuales por horizonte).
- Para una predicción p, se genera un conjunto de muestras p + residual_bootstrap (n_boot muestras) y se extraen los quantiles 2.5% y 97.5% para construir `lower_95` y `upper_95`.
- Implementación: función `bootstrap_intervalos(predicciones, residuales, alpha=0.05, n_boot=1000, random_state=...)`.

### Baseline y métricas
- Baseline: persistencia simple usando `grco2_kwh_lag_1h` como predicción para cualquier horizonte (se evalúa este baseline con Y_test).
- Métricas calculadas por horizonte: `MAE` y `RMSE` para modelo y baseline.
- Se calcula `mejora_% = (1 - MAE_modelo / MAE_baseline) * 100`.

---

## Resultados

### Métricas por horizonte (extracto)
- Horizonte 1: MAE modelo = 4.8145, RMSE modelo = 6.4001. Baseline MAE = 6.9026, RMSE = 8.6993. Mejora: 30.25%.
- Horizonte 6: MAE modelo = 9.3509, RMSE = 11.9173. Baseline MAE = 17.5737, RMSE = 20.6214. Mejora: 46.79%.
- Horizonte 24: MAE modelo = 13.8869, RMSE modelo = 17.3229. Baseline MAE = 11.2504, RMSE = 15.3383. Mejora: -23.43% (el baseline gana en este tramo en la primera ejecución; revisar estabilidad de horizontes largos).
- Horizonte 48: MAE modelo = 14.5647, RMSE modelo = 17.5949. Baseline MAE = 14.9625, RMSE = 19.0548. Mejora: 2.66%.

(La tabla completa se encuentra en `comparacion_terminal.csv` generada por `modeling.py`.)

### Muestra de pronóstico 48h (primeras filas)
| fecha_predicha       | horizonte | prediccion | lower_95  | upper_95  |
|----------------------|----------:|-----------:|----------:|----------:|
| 2025-12-30 00:00:00  |         1 | 154.766800 | 143.553819| 170.785702|
| 2025-12-30 01:00:00  |         2 | 152.389816 | 138.637532| 171.331439|
| 2025-12-30 02:00:00  |         3 | 152.456589 | 137.216954| 173.623241|
| 2025-12-30 03:00:00  |         4 | 139.659210 | 121.320246| 161.367526|
| 2025-12-30 04:00:00  |         5 | 126.853714 | 108.515389| 149.966894|

### Observaciones
- El modelo supera consistentemente el baseline en horizontes cortos (1-12h).
- En horizontes intermedios/largos (>24h) el rendimiento es mixto: en ocasiones el baseline de persistencia es competitivo o mejor. Posibles causas: calidad/representatividad de features exógenas, estrategia de asignación de hiperparámetros por rangos, o dependencia estacional no capturada.
- Las bandas bootstrap crecen con el horizonte, indicando mayor incertidumbre en predicciones lejanas.

### Espacio para imágenes (placeholders)
- Gráfico de validación (real vs predicción + baseline + IC): `validation_vs_model.png` (incluir imagen aquí)

- Gráfico del pronóstico 48h con bandas: `forecast_48h.png` (incluir imagen aquí)

- Gráficas generadas desde CSV (opcionales): `validation_vs_model_from_csv.png`, `forecast_48h_from_csv.png` (incluir cuando estén disponibles)

---

## Artefactos generados por los scripts
- `modeling.py` produce:
  - `forecast_48h.csv` — pronóstico con `lower_95`/`upper_95`.
  - `comparacion_terminal.csv` — métricas por horizonte.
  - `validation_vs_model.png`, `forecast_48h.png` — gráficos.
- `export_results.py` produce:
  - `validation_vs_model_from_csv.png`, `forecast_48h_from_csv.png`.

---

## Reproducción: paso a paso (comandos)

1) Preparación del entorno (macOS + pyenv):

```bash
HOMEBREW_NO_AUTO_UPDATE=1 brew install libomp
pyenv shell .envMaestria
pip install pandas numpy scikit-learn xgboost matplotlib joblib
```

2) Ejecutar pipeline completo (entrenamiento, validación, predicción y exportación):

```bash
/Users/sruiz.gomez/.pyenv/versions/.envMaestria/bin/python modeling.py
```

3) Regenerar gráficos desde CSV sin re-entrenar:

```bash
/Users/sruiz.gomez/.pyenv/versions/.envMaestria/bin/python export_results.py
```

---

## Uso de modelos y cómo obtener resultados

### Exportar resultados sin re-entrenar
- `export_results.py` carga `forecast_48h.csv` y `comparacion_terminal.csv` (generados por `modeling.py`) y regenera PNGs y un resumen rápido.

### Serialización de modelos (opcional, recomendado)
- Para producción y pronósticos rápidos recomendamos serializar cada modelo por horizonte con `joblib`:

```python
from joblib import dump, load
# Guardar tras entrenar modelo 'model' para el horizonte h
dump(model, f'model_h{h}.joblib')

# Cargar más tarde
model = load('model_h1.joblib')
# Predecir
pred = model.predict(X_row)
```

- Además de los modelos, recomendamos guardar `residuales_por_horizonte[h]` (por ejemplo en `residuales_h{h}.npy`) para poder reconstruir IC por bootstrap sin acceder a los folds OOS.

### Pronóstico desde modelos guardados (flujo recomendado)
1. Entrenar y serializar: ejecutar `modeling.py` modificado para guardar `model_h{h}.joblib` y `residuales_h{h}.npy`.
2. Crear `predict_from_models.py` que:
   - Cargue modelos serializados y residuales.
   - Construya la fila de features `X_row` para la fecha de inicio (usar idénticas transformaciones de `crear_features`).
   - Para cada horizonte `h`, calcule `pred_h = model_h.predict(X_row)` y calcule `lower_95`/`upper_95` usando `residuales_h{h}.npy` con la función `bootstrap_intervalos`.
   - Escriba `forecast_48h.csv` con `fecha_predicha`, `horizonte`, `prediccion`, `lower_95`, `upper_95`.

> Nota: si quieres, implemento `predict_from_models.py` y modifico `modeling.py` para que guarde automáticamente los modelos y residuales durante el entrenamiento.

---

## Solución de problemas comunes
- Error XGBoost `libxgboost.dylib could not be loaded`: instala `libomp` y asegúrate de usar el intérprete del entorno.
- Error por columnas `datetime64[ns]` en XGBoost: filtra `X` a tipos numéricos y booleanos (`X.select_dtypes(include=[np.number, "bool"])`).
- `mean_squared_error(..., squared=False)` no soportado en versiones antiguas: usar `np.sqrt(mean_squared_error(...))`.
- Parquet no encontrado: verifica el path y permisos; el script imprime el esquema de columnas detectadas.

---

## Recomendaciones y siguientes pasos
- Serializar modelos (`joblib`) y residuales (`npy`) para producción y pronósticos rápidos.
- Añadir `requirements.txt` o `pyproject.toml` para control de dependencias y reproducibilidad.
- Evaluar calibración empírica de los intervalos (coverage test 95% en `Y_test`).
- Explorar feature importance por horizonte y refinar la selección de features por rango de horizonte.
- Considerar modelos probabilísticos o quantile regression (LightGBM quantile, NGBoost, etc.) como alternativa al bootstrap.
- Organizar salidas en `results/` para mantener el repo ordenado y versionado.

---

Si quieres, aplico ahora alguna de estas tareas: serializar los modelos con `joblib` y añadir `predict_from_models.py`; crear `requirements.txt`; calcular cobertura empírica y/o mover artefactos a `results/`.
