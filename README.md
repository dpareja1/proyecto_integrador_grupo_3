# ⚡ Predicción del Precio de Bolsa — Mercado Eléctrico Colombiano

> **Proyecto Integrador — Maestría en Ciencia de Datos y Analítica**  
> Universidad EAFIT · 2026

---

## 👥 Autores

| Nombre | 
|--------|
| Daniel Pareja |
| Juan José Morales |
| Sebastián Ruiz |
| Santiago Molano |
| David Gómez |

---

## 📌 Descripción

Este proyecto desarrolla un pipeline completo de ciencia de datos orientado a la predicción 
del precio de bolsa (COP/kWh) en el mercado eléctrico colombiano con un horizonte de 24 horas, 
integrando análisis de emisiones de CO₂, dinámica de participación térmica y evaluación 
comparativa de modelos de aprendizaje automático.

El resultado final es un **dashboard interactivo** construido en Streamlit que permite explorar 
los datos, comparar modelos, analizar el rendimiento de las predicciones y contextualizar los 
resultados dentro de eventos climáticos relevantes como el Fenómeno del Niño 2023–2024.

---

## 🎯 Motivación

La demanda eléctrica en Colombia presenta alta variabilidad horaria. En momentos de pico, 
el sistema recurre a generación térmica, incrementando las emisiones de CO₂ y el precio de bolsa. 
Este fenómeno se intensifica durante períodos de estrés hídrico como el Fenómeno del Niño, 
donde la menor disponibilidad hidráulica presiona al alza tanto los precios como la intensidad 
de emisiones.

Anticipar el precio de bolsa con 24 horas de antelación permite:
- Redistribuir la demanda hacia ventanas horarias de menor intensidad de emisiones
- Apoyar la formulación de políticas energéticas más sostenibles
- Mejorar la eficiencia operativa del sistema eléctrico

---

## 📊 Estructura del Dashboard

| Pestaña | Contenido |
|---------|-----------|
| 🏠 **Inicio** | Portada con KPIs globales e índice de secciones |
| 🌿 **Emisiones CO₂** | Medidor horario de intensidad de emisiones con alertas por nivel de riesgo |
| 📊 **Precio Bolsa vs Participación Térmica** | Series temporales comparadas con sombreado del Fenómeno del Niño |
| 📊 **Comparación de Modelos** | Tabla y gráfico comparativo de métricas por modelo |
| 🤖 **Rendimiento del Modelo** | Predicción vs. real, banda de error, intervalo de confianza y ventanas Q1/Q3 |
| 📋 **Metodología** | Marco teórico, fuentes de datos, métricas y referencias |

---

## 🤖 Modelos Evaluados

| Modelo | Descripción |
|--------|-------------|
| Original (47 feat, recursivo 48h) | Baseline con conjunto completo de variables |
| Simplificado (15 feat, recursivo 24h) | Reducción de dimensionalidad para mayor generalización |
| DMS (24 modelos independientes) | Direct Multi-Step: un modelo por cada hora del horizonte |
| **HYBRID (Baseline + XGBoost)** | **Mejor modelo** — combinación de predictor base con boosting |

---

## 📁 Estructura del Repositorio

├── dashboard.py                          # Aplicación principal Streamlit
├── requirements.txt                      # Dependencias del proyecto
├── README.md                             # Este archivo
└── data/
├── aporte_hidricos_agrupados.parquet
├── demanda_comercial.parquet
├── generacion_completa.parquet
├── generacion_emision_agrupada.parquet
├── intensidad_Co2.parquet
├── grafico.parquet
├── resultados.csv
└── resultados_modelos.csv

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/<nombre-repo>.git
cd <nombre-repo>
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ubicar los archivos de datos

Asegúrate de que todos los archivos `.parquet` y `.csv` estén en la **misma carpeta** 
que `dashboard.py`, o dentro de la carpeta `data/` si así lo configuras.

### 4. Ejecutar el dashboard

```bash
python -m streamlit run dashboard.py
```

El dashboard abrirá automáticamente en `http://localhost:8501`

---

## 📦 Dependencias principales

streamlit
plotly
pandas
numpy
pyarrow

Archivo completo en `requirements.txt`.

---

## 📈 Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error Absoluto Medio — magnitud promedio del error en COP/kWh |
| **RMSE** | Raíz del Error Cuadrático Medio — penaliza errores grandes (picos de precio) |
| **R²** | Coeficiente de Determinación — proporción de varianza explicada por el modelo |
| **Cobertura IC** | % de observaciones reales dentro del intervalo de confianza estimado |

## Extracción y almacenamiento de datos
`proyecto_integrador` contiene la ingesta y el componente de analitica/ML del proyecto. Desde este repositorio se consumen los datos publicos de SIMEM, se almacenan en Amazon S3 en la capa `bronze` y se define la tuberia de entrenamiento en Amazon SageMaker para construir y evaluar modelos de prediccion sobre las tablas refinadas.

`proyecto_integrador_glue_pipeline` contiene la capa de ingenieria de datos en AWS Glue. Desde este repositorio se despliegan y ejecutan los jobs que transforman los archivos JSON crudos desde `bronze` hacia `silver`, generan la capa `gold` de features para modelado y actualizan los crawlers y catalogos necesarios para su consumo en Athena y otras herramientas analiticas.

La arquitectura general, el flujo de datos y los `GitHub Actions` asociados se documentan en el diagrama ![Diagrama de arquitectura](arquitectura_integrador_aws).

1. [santfirax/proyecto_integrador_glue_pipeline](https://github.com/santfirax/proyecto_integrador_glue_pipeline): despliegue y ejecucion de pipelines AWS Glue para capas silver y gold.
2. [santfirax/proyecto_integrador](https://github.com/santfirax/proyecto_integrador): ingesta de datos SIMEM y pipeline de entrenamiento en SageMaker.

---

## ⚠️ Limitaciones

- No incorpora decisiones regulatorias ni cambios de política energética de forma explícita
- Variables climatológicas de alta resolución espacial no están incluidas en la versión actual
- El horizonte de predicción está limitado a 24 horas

---

## 🔭 Trabajo Futuro

- Incorporación de variables exógenas climáticas (índices ENSO, precipitación regional)
- Exploración de arquitecturas LSTM y Transformers para series temporales
- Extensión del horizonte de predicción a 48–72 horas
- Evaluación de transferibilidad a otros mercados eléctricos de la región

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos como entrega final del 
**Proyecto Integrador** de la Maestría en Ciencia de Datos y Analítica de la 
Universidad EAFIT. No está destinado a uso comercial.


---

<p align="center">
  Universidad EAFIT · Maestría en Ciencia de Datos y Analítica · 2026
</p>
