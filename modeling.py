import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_parquet('part-00000-4f2b0623-696a-42ac-a276-7ba8ad7618ad-c000.snappy.parquet')

# ============================================================
# 1. Preparar datos
# ============================================================

df['dateh'] = pd.to_datetime(df['dateh'], format='%Y-%m-%dT%H:%M:%S')   
df = df.sort_values("dateh").set_index("dateh")

target = "grco2_kwh"
max_horizon = 48


FEATURES_YA_EN_PARQUET = [
    "year",
    "month",
    "day",
    "dayofweek",
    "quarter",
    "week",
    "grco2_kwh_lag_1h",
    "grco2_kwh_lag_2h",
    "grco2_kwh_lag_3h",
    "grco2_kwh_lag_4h",
    "grco2_kwh_lag_5h",
    "grco2_kwh_lag_6h",
    "grco2_kwh_lag_7h",
    "grco2_kwh_lag_8h",
    "grco2_kwh_lag_9h",
    "grco2_kwh_lag_10h",
    "grco2_kwh_lag_11h",
    "grco2_kwh_lag_12h",
    "grco2_kwh_lag_13h",
    "grco2_kwh_lag_14h",
    "grco2_kwh_lag_15h",
    "grco2_kwh_lag_16h",
    "grco2_kwh_lag_17h",
    "grco2_kwh_lag_18h",
    "grco2_kwh_lag_19h",
    "grco2_kwh_lag_20h",
    "grco2_kwh_lag_21h",
    "grco2_kwh_lag_22h",
    "grco2_kwh_lag_23h",
    "grco2_kwh_lag_24h",
    "grco2_kwh_avg_1d",
    "grco2_kwh_avg_2d",
    "grco2_kwh_avg_3d",
    "grco2_kwh_trend",
    "total_aportes_lag1",
    "total_aportes_lag7",
    "total_aportes_lag30",
    "total_aportes_avg7d",
    "total_aportes_avg30d",
    "es_festivo",
    "es_fin_semana",
    "es_festivo_largo",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "quarter_sin",
    "quarter_cos",
    "week_sin",
    "week_cos",
    "dateh_ts",
]

FEATURES_NUEVAS_A_CREAR = [
    "hour",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "lag_48",
    "lag_72",
    "lag_168",
    "roll_std_24",
    "roll_min_24",
    "roll_max_24",
    "roll_std_48",
    "roll_min_48",
    "roll_max_48",
    "roll_std_72",
    "roll_min_72",
    "roll_max_72",
    "roll_std_168",
    "roll_min_168",
    "roll_max_168",
]


# ============================================================
# 2. Crear features
# ============================================================

def crear_features(df, target="grco2_kwh"):
    df = df.copy()

    # Variables calendario nuevas que no vienen en el parquet.
    df["hour"] = df.index.hour
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Codificación cíclica de la hora del día.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Lags largos y estadísticos móviles para complementar lo que ya trae el parquet.
    lags = [48, 72, 168]
    y_past = df[target].shift(1)

    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)

    windows = [24, 48, 72, 168]

    for w in windows:
        df[f"roll_std_{w}"] = y_past.rolling(w).std()
        df[f"roll_min_{w}"] = y_past.rolling(w).min()
        df[f"roll_max_{w}"] = y_past.rolling(w).max()

    return df


print("=== FEATURES YA PRESENTES EN EL PARQUET ===")
print(", ".join(FEATURES_YA_EN_PARQUET))
print("=== NUEVAS FEATURES QUE SE CREARAN ===")
print(", ".join(FEATURES_NUEVAS_A_CREAR))


# ============================================================
# 3. Crear targets de las próximas 48 horas
# ============================================================

def crear_targets(df, target="grco2_kwh", max_horizon=48):
    df = df.copy()

    for h in range(1, max_horizon + 1):
        df[f"target_h{h}"] = df[target].shift(-h)

    return df


def bootstrap_intervalos(predicciones, residuales, alpha=0.05, n_boot=1000, random_state=42):
    predicciones = np.asarray(predicciones)
    residuales = np.asarray(residuales)

    if residuales.size == 0:
        raise ValueError("No hay residuales disponibles para calcular intervalos bootstrap.")

    rng = np.random.default_rng(random_state)
    muestras_residuales = rng.choice(
        residuales,
        size=(n_boot, predicciones.size),
        replace=True,
    )
    muestras_pred = predicciones.reshape(1, -1) + muestras_residuales

    limite_inferior = np.quantile(muestras_pred, alpha / 2, axis=0)
    limite_superior = np.quantile(muestras_pred, 1 - alpha / 2, axis=0)

    return limite_inferior, limite_superior


df_feat = crear_features(df, target=target)
df_model = crear_targets(df_feat, target=target, max_horizon=max_horizon)

df_model = df_model.dropna()


# ============================================================
# 4. Separar X e Y
# ============================================================

target_cols = [f"target_h{h}" for h in range(1, max_horizon + 1)]

X = df_model.drop(columns=[target] + target_cols)
Y = df_model[target_cols]

X = X.select_dtypes(include=[np.number, "bool"]).copy()

# ============================================================
# 5. Separar train y test temporal
# ============================================================

test_size = 30 * 24  # últimos 30 días para test

X_train = X.iloc[:-test_size]
Y_train = Y.iloc[:-test_size]

X_test = X.iloc[-test_size:]
Y_test = Y.iloc[-test_size:]

print("=== RANGOS TEMPORALES ===")
print(f"Entrenamiento: {X_train.index.min()} -> {X_train.index.max()} ({len(X_train)} filas)")
print(f"Test:          {X_test.index.min()} -> {X_test.index.max()} ({len(X_test)} filas)")


# ============================================================
# 6. Buscar mejores parámetros en horizontes clave
# ============================================================

param_grid = {
    "n_estimators": [300, 500, 800, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.5],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 5, 10]
}

tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=7 * 24  # cada validación evalúa 7 días
)

print("=== VALIDACION TEMPORAL (TimeSeriesSplit) ===")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    train_start = X_train.index[train_idx[0]]
    train_end = X_train.index[train_idx[-1]]
    val_start = X_train.index[val_idx[0]]
    val_end = X_train.index[val_idx[-1]]
    print(
        f"Fold {fold}: train {train_start} -> {train_end} ({len(train_idx)} filas) | "
        f"validacion {val_start} -> {val_end} ({len(val_idx)} filas)"
    )

horizontes_clave = [1, 6, 12, 24, 48]

best_params_by_horizon = {}

for h in horizontes_clave:
    print(f"\nOptimizando horizonte +{h}h")

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, Y_train[f"target_h{h}"])

    best_params_by_horizon[h] = search.best_params_

    print("Mejores parámetros:")
    print(search.best_params_)


# ============================================================
# 7. Asignar parámetros según horizonte
# ============================================================

def seleccionar_params(h):
    if h <= 3:
        return best_params_by_horizon[1]
    elif h <= 9:
        return best_params_by_horizon[6]
    elif h <= 18:
        return best_params_by_horizon[12]
    elif h <= 36:
        return best_params_by_horizon[24]
    else:
        return best_params_by_horizon[48]


# ============================================================
# 8. Entrenar 48 modelos
# ============================================================

modelos = {}

for h in range(1, max_horizon + 1):
    print(f"Entrenando modelo horizonte +{h}h")

    params = seleccionar_params(h)

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **params
    )

    model.fit(
        X_train,
        Y_train[f"target_h{h}"]
    )

    modelos[h] = model


# ============================================================
# 8.1. Residuales fuera de muestra para intervalos bootstrap
# ============================================================

residuales_por_horizonte = {}

for h in range(1, max_horizon + 1):
    params = seleccionar_params(h)
    residuales_fold = []

    for train_idx, val_idx in tscv.split(X_train):
        fold_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params,
        )

        fold_model.fit(
            X_train.iloc[train_idx],
            Y_train[f"target_h{h}"].iloc[train_idx],
        )

        val_pred = fold_model.predict(X_train.iloc[val_idx])
        residuales_fold.extend(
            Y_train[f"target_h{h}"].iloc[val_idx].values - val_pred
        )

    residuales_por_horizonte[h] = np.asarray(residuales_fold)
    print(
        f"Residuales OOS +{h}h: {len(residuales_por_horizonte[h])} observaciones"
    )


# ============================================================
# 9. Evaluación por horizonte
# ============================================================

resultados = []

for h in range(1, max_horizon + 1):
    y_real = Y_test[f"target_h{h}"]
    y_pred = modelos[h].predict(X_test)

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))

    resultados.append({
        "horizonte": h,
        "MAE": mae,
        "RMSE": rmse
    })

resultados = pd.DataFrame(resultados)

print(resultados)


# ============================================================
# 10. Baseline de persistencia
# ============================================================

baseline_resultados = []

for h in range(1, max_horizon + 1):
    y_real = Y_test[f"target_h{h}"]

    # Baseline simple: usar último valor conocido
    y_baseline = X_test["grco2_kwh_lag_1h"]

    mae_base = mean_absolute_error(y_real, y_baseline)
    rmse_base = np.sqrt(mean_squared_error(y_real, y_baseline))

    baseline_resultados.append({
        "horizonte": h,
        "MAE_baseline": mae_base,
        "RMSE_baseline": rmse_base,
    })

baseline_resultados = pd.DataFrame(baseline_resultados)

comparacion = resultados.merge(
    baseline_resultados,
    on="horizonte"
)

comparacion["mejora_%"] = (
    1 - comparacion["MAE"] / comparacion["MAE_baseline"]
) * 100

print(comparacion)


# ============================================================
# 11. Pronosticar las próximas 48 horas
# ============================================================

ultima_fila = X.iloc[[-1]]
ultima_fecha = X.index[-1]

forecast = []
forecast_lower = []
forecast_upper = []

for h in range(1, max_horizon + 1):
    pred = modelos[h].predict(ultima_fila)[0]
    limite_inferior, limite_superior = bootstrap_intervalos(
        np.array([pred]),
        residuales_por_horizonte[h],
        alpha=0.05,
        n_boot=1000,
        random_state=42 + h,
    )

    forecast.append({
        "fecha_predicha": ultima_fecha + pd.Timedelta(hours=h),
        "horizonte": h,
        "prediccion": pred
    })
    forecast_lower.append(limite_inferior[0])
    forecast_upper.append(limite_superior[0])

forecast_48h = pd.DataFrame(forecast)
forecast_48h["lower_95"] = forecast_lower
forecast_48h["upper_95"] = forecast_upper

print(forecast_48h)
# Export forecast to CSV
forecast_48h.to_csv('forecast_48h.csv', index=False)


# ============================================================
# 11.1 Comparación real vs predicción en terminal
# ============================================================

comparacion_terminal = pd.DataFrame({
    "fecha": X_test.index,
    "real_+1h": Y_test["target_h1"].values,
    "pred_+1h": modelos[1].predict(X_test),
    "baseline_+1h": X_test["grco2_kwh_lag_1h"].values,
})

limite_inferior_1h, limite_superior_1h = bootstrap_intervalos(
    comparacion_terminal["pred_+1h"].values,
    residuales_por_horizonte[1],
    alpha=0.05,
    n_boot=1000,
    random_state=42,
)

comparacion_terminal["lower_95"] = limite_inferior_1h
comparacion_terminal["upper_95"] = limite_superior_1h

comparacion_terminal["error_abs"] = (
    comparacion_terminal["real_+1h"] - comparacion_terminal["pred_+1h"]
).abs()

print(comparacion_terminal.head(20))
print(
    "Resumen +1h -> "
    f"MAE: {mean_absolute_error(comparacion_terminal['real_+1h'], comparacion_terminal['pred_+1h']):.4f}, "
    f"RMSE: {np.sqrt(mean_squared_error(comparacion_terminal['real_+1h'], comparacion_terminal['pred_+1h'])):.4f}"
)
print(
    "Baseline +1h -> "
    f"MAE: {mean_absolute_error(comparacion_terminal['real_+1h'], comparacion_terminal['baseline_+1h']):.4f}, "
    f"RMSE: {np.sqrt(mean_squared_error(comparacion_terminal['real_+1h'], comparacion_terminal['baseline_+1h'])):.4f}"
)

# Export comparison to CSV
comparacion_terminal.to_csv('comparacion_terminal.csv', index=False)


# ============================================================
# 12. Graficar
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(
    X_test.index,
    Y_test["target_h1"],
    label="Real (+1h)",
    linewidth=2
)
plt.plot(
    X_test.index,
    modelos[1].predict(X_test),
    label="Predicción (+1h)",
    linewidth=2,
    alpha=0.8
)
plt.plot(
    X_test.index,
    comparacion_terminal["baseline_+1h"],
    label="Baseline (+1h)",
    linewidth=2,
    linestyle="--",
    color="tab:red",
    alpha=0.9,
)
plt.fill_between(
    X_test.index,
    comparacion_terminal["lower_95"],
    comparacion_terminal["upper_95"],
    color="tab:blue",
    alpha=0.15,
    label="IC bootstrap 95%",
)
plt.title("Comparación real vs predicción en el test (+1h)")
plt.xlabel("Fecha")
plt.ylabel("grco2_kwh")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('validation_vs_model.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(
    forecast_48h["fecha_predicha"],
    forecast_48h["prediccion"],
    marker="o"
)
plt.fill_between(
    forecast_48h["fecha_predicha"],
    forecast_48h["lower_95"],
    forecast_48h["upper_95"],
    color="tab:orange",
    alpha=0.18,
    label="IC bootstrap 95%",
)
plt.title("Pronóstico próximas 48 horas")
plt.xlabel("Fecha")
plt.ylabel("Predicción")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.savefig('forecast_48h.png', bbox_inches='tight')
plt.show()