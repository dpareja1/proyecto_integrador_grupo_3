# ============================================================
# DASHBOARD ENERGÉTICO - NIVEL MAESTRÍA
# Ejecutar desde la misma carpeta donde están los archivos:
#   generacion_emision_agrupada.parquet
#   grafico.parquet Valor
#   resultados.csv
# paper
# Comando: streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURACIÓN GLOBAL DE FUENTES  ← MODIFICA AQUÍ LOS TAMAÑOS
# ============================================================
FONT_TITLE_MAIN    = 28
FONT_SECTION_TITLE = 22
FONT_AXIS_TITLE    = 16
FONT_AXIS_TICK     = 14
FONT_LEGEND        = 14
FONT_ANNOTATION    = 13
FONT_FENOMENO_NINO = 10
FONT_GAUGE_NUMBER  = 20
FONT_GAUGE_TICK    = 13
FONT_SIDEBAR       = 14
FONT_METRIC_LABEL  = 13
# ============================================================

# ============================================================
# RANGOS DEL MEDIDOR grCO2/kWh  ← MODIFICA AQUÍ LOS UMBRALES
# ============================================================
GAUGE_MIN          = 0
GAUGE_MAX          = 600
RANGE_VERDE_MAX    = 150
RANGE_AMARILLO_MAX = 300
RANGE_NARANJA_MAX  = 450
# ============================================================

# ============================================================
# PERIODO FENÓMENO DEL NIÑO  ← MODIFICA AQUÍ LAS FECHAS
# ============================================================
NINO_START = "2023-12-01"
NINO_END   = "2024-04-30"
# ============================================================

# ---- Configuración de página ----
st.set_page_config(
    page_title="Dashboard Energético",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    .stApp {{ background-color: #F7F9FC; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 1.5rem; }}
    h1 {{ font-size: {FONT_TITLE_MAIN}px !important; color: #1a2e4a; font-weight: 700; }}
    h2 {{ font-size: {FONT_SECTION_TITLE}px !important; color: #1a2e4a; }}
    h3 {{ font-size: {FONT_SECTION_TITLE - 2}px !important; color: #2c4a6e; }}
    .css-1d391kg, [data-testid="stSidebar"] {{ background-color: #EEF2F7; }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] p {{ font-size: {FONT_SIDEBAR}px !important; }}
    hr {{ border-color: #CBD5E0; margin: 1.2rem 0; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CARGA DE DATOS
# ============================================================
@st.cache_data(ttl=0)
def cargar_datos():
    df_emision   = pd.read_parquet("generacion_emision_agrupada.parquet")
    df_grafico   = pd.read_parquet("grafico.parquet")
    df_resultado = pd.read_csv("resultados.csv")
    return df_emision, df_grafico, df_resultado

df_emision, df_grafico, df_resultado = cargar_datos()

# ---- Pre-procesamiento: Emisiones ----
df_emision["DateH"] = pd.to_datetime(df_emision["DateH"])
df_emision["Fecha"] = df_emision["DateH"].dt.date
df_emision["Hora"]  = df_emision["DateH"].dt.hour

# ---- Pre-procesamiento: Gráfico mercado (sin groupby — ya viene procesado) ----
df_grafico["Fecha"] = pd.to_datetime(
    df_grafico["year"].astype(str) + "-" + df_grafico["month"].astype(str).str.zfill(2)
)
df_grafico = df_grafico.sort_values("Fecha")

# ---- Pre-procesamiento: Resultado modelo ----
df_resultado["dateh"] = pd.to_datetime(df_resultado["dateh"])
df_resultado = df_resultado.sort_values("dateh")


# ============================================================
# ENCABEZADO
# ============================================================
st.markdown("# ⚡ Dashboard Energético — Sector Eléctrico Colombiano")
st.markdown("---")

tab0, tab1, tab2, tab2b, tab3, tab4 = st.tabs([
    "🏠 Inicio",
    "🌿 Emisiones CO₂",
    "📊 Precio Bolsa vs Participación Térmica",
    "📊 Comparación de Modelos",
    "🤖 Rendimiento del Modelo",
    "📋 Metodología"
])

# ============================================================
# PESTAÑA 0 — PORTADA / LOBBY
# ============================================================
with tab0:

    total_registros   = len(df_emision)
    modelos_evaluados = df_resultado["modelo"].nunique()
    mejor_modelo      = df_resultado.groupby("modelo")["error_abs"].mean().idxmin()
    mae_mejor         = df_resultado.groupby("modelo")["error_abs"].mean().min()
    rango_inicio      = df_emision["DateH"].min().strftime("%b %Y")
    rango_fin         = df_emision["DateH"].max().strftime("%b %Y")

    MESES_PORTADA = {
        "January":"enero","February":"febrero","March":"marzo","April":"abril",
        "May":"mayo","June":"junio","July":"julio","August":"agosto",
        "September":"septiembre","October":"octubre","November":"noviembre","December":"diciembre"
    }
    hoy_raw = pd.Timestamp.today().strftime("%d de %B de %Y")
    for en, es in MESES_PORTADA.items():
        hoy_raw = hoy_raw.replace(en, es)
    hoy = hoy_raw
    
    autores_html = "".join([
        f"<span style='background:rgba(255,255,255,0.13); border:1px solid rgba(255,255,255,0.22); "
        f"border-radius:20px; padding:6px 18px; font-size:{FONT_METRIC_LABEL + 1}px; "
        f"color:#FFFFFF; font-weight:500;'>{n}</span>"
        for n in ["Daniel Pareja","Juan José Morales","Sebastián Ruiz","Santiago Molano","David Gómez"]
    ])

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a2e4a 0%,#2c4a6e 60%,#1a6b8a 100%);
                border-radius:18px; padding:56px 60px 44px 60px; margin-bottom:32px;
                box-shadow:0 8px 32px rgba(26,46,74,0.18);'>
        <div style='display:flex; align-items:center; gap:16px; margin-bottom:28px;'>
            <div style='background:rgba(255,255,255,0.12); border-radius:12px; padding:10px 20px;
                        font-size:13px; color:rgba(255,255,255,0.75); letter-spacing:2px;
                        font-weight:600; text-transform:uppercase;'>
                Universidad EAFIT &nbsp;·&nbsp; Maestría en Ciencia de Datos
            </div>
        </div>
        <h1 style='font-size:42px; font-weight:800; color:#FFFFFF;
                   margin:0 0 10px 0; line-height:1.15; letter-spacing:-0.5px;'>
            Predicción de Emisiones<br>en el Mercado Eléctrico Colombiano
        </h1>
        <p style='font-size:18px; color:rgba(255,255,255,0.75); margin:0 0 36px 0;
                  font-weight:400; line-height:1.5;'>
            Análisis de emisiones, dinámica de mercado y evaluación de modelos<br>
            de aprendizaje automático para horizontes de 24 horas
        </p>
        <div style='border-top:1px solid rgba(255,255,255,0.2); margin-bottom:32px;'></div>
        <div style='margin-bottom:28px;'>
            <p style='font-size:12px; color:rgba(255,255,255,0.55); letter-spacing:2px;
                      text-transform:uppercase; margin:0 0 10px 0; font-weight:600;'>Autores</p>
            <div style='display:flex; flex-wrap:wrap; gap:10px;'>{autores_html}</div>
        </div>
        <p style='font-size:13px; color:rgba(255,255,255,0.5); margin:0; letter-spacing:1px;'>
            📅 {hoy}
        </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "⚡", "Registros de Generación", f"{total_registros:,}",      "#2980B9"),
        (k2, "📅", "Período Analizado",        f"{rango_inicio} – {rango_fin}", "#8E44AD"),
        (k3, "🤖", "Modelos Evaluados",        str(modelos_evaluados),      "#27AE60"),
        (k4, "🏆", "Mejor Modelo",             mejor_modelo,                "#E67E22"),
        (k5, "📉", "MAE Mejor Modelo",         f"{mae_mejor:.3f}",          "#E74C3C"),
    ]
    for col, icono, label, valor, color in kpis:
        col.markdown(f"""
        <div style='background:#FFFFFF; border-radius:14px; padding:22px 16px;
                    border-top:4px solid {color}; text-align:center;
                    box-shadow:0 2px 10px rgba(0,0,0,0.07);'>
            <div style='font-size:26px; margin-bottom:6px;'>{icono}</div>
            <p style='font-size:{FONT_METRIC_LABEL}px; color:#888; margin:0 0 6px 0;'>{label}</p>
            <p style='font-size:{FONT_SECTION_TITLE - 2}px; font-weight:700;
                      color:{color}; margin:0; word-break:break-word;'>{valor}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{FONT_SECTION_TITLE - 4}px; font-weight:700; "
                f"color:#1a2e4a; margin-bottom:14px;'>📑 Contenido del Dashboard</p>",
                unsafe_allow_html=True)

    secciones = [
        ("🌿", "Emisiones CO₂",
         "Medidor de intensidad de emisiones horarias con alertas por nivel de riesgo ambiental."),
        ("📊", "Precio Bolsa vs Participación Térmica",
         "Evolución del precio de bolsa y la participación térmica con análisis del Fenómeno del Niño 2024."),
        ("📊", "Comparación de Modelos",
         "Tabla y gráfico comparativo de todos los modelos evaluados con sus métricas de desempeño."),
        ("🤖", "Rendimiento del Modelo",
         "Análisis detallado de predicción vs. valor real, banda de error e intervalo de confianza."),
        ("📋", "Metodología",
         "Marco teórico, fuentes de datos, descripción del modelo, métricas y referencias bibliográficas."),
    ]

    for icono, titulo, desc in secciones:
        st.markdown(f"""
        <div style='background:#FFFFFF; border-radius:12px; padding:16px 20px; margin-bottom:10px;
                    display:flex; align-items:flex-start; gap:16px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.06); border-left:4px solid #2980B9;'>
            <span style='font-size:22px; line-height:1;'>{icono}</span>
            <div>
                <p style='font-size:{FONT_METRIC_LABEL + 2}px; font-weight:700;
                          color:#1a2e4a; margin:0 0 4px 0;'>{titulo}</p>
                <p style='font-size:{FONT_METRIC_LABEL}px; color:#666; margin:0;'>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:28px; text-align:center;'>
        <p style='font-size:{FONT_METRIC_LABEL}px; color:#AAB4C0;'>
            Universidad EAFIT · Maestría en Ciencia de Datos · {hoy}
        </p>
    </div>
    """, unsafe_allow_html=True)
# ============================================================
# PESTAÑA 1 — MEDIDOR DE EMISIONES
# ============================================================
with tab1:
    st.markdown("## 🌿 Intensidad de Emisiones de CO₂")

    # ---- Filtros — solo visibles en esta pestaña ----
    fechas_disponibles = sorted(df_emision["Fecha"].unique())
    fechas_dt = pd.to_datetime(fechas_disponibles)

    años_disp  = sorted(fechas_dt.year.unique())
    col_a, col_m, col_d, col_h = st.columns(4)

    with col_a:
        año_sel = st.selectbox("📅 Año:", options=años_disp, index=len(años_disp) - 1)

    meses_disp = sorted(fechas_dt[fechas_dt.year == año_sel].month.unique())
    MESES_ES   = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril",
                  5:"Mayo",  6:"Junio",   7:"Julio", 8:"Agosto",
                  9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    with col_m:
        mes_sel = st.selectbox(
            "🗓️ Mes:",
            options=meses_disp,
            format_func=lambda m: MESES_ES[m],
            index=len(meses_disp) - 1
        )

    dias_disp = sorted(fechas_dt[(fechas_dt.year == año_sel) & (fechas_dt.month == mes_sel)].day.unique())
    with col_d:
        dia_sel = st.selectbox("📆 Día:", options=dias_disp, index=len(dias_disp) - 1)

    import datetime
    fecha_sel = datetime.date(año_sel, mes_sel, dia_sel)

    with col_h:
        horas_disponibles = sorted(df_emision[df_emision["Fecha"] == fecha_sel]["Hora"].unique())
        hora_sel = st.selectbox("🕐 Hora:", options=horas_disponibles, index=len(horas_disponibles) - 1)
    fila_sel = df_emision[
        (df_emision["Fecha"] == fecha_sel) & (df_emision["Hora"] == hora_sel)
    ]

    if fila_sel.empty:
        st.warning("No hay datos para la fecha y hora seleccionadas.")
        valor_co2 = 0
        gen_mwh   = 0
    else:
        valor_co2 = fila_sel["grCO2/KWh"].values[0]
        gen_mwh   = fila_sel["Generacion[MWh]"].values[0]

    def clasificar_co2(v):
        if v <= RANGE_VERDE_MAX:
            return "🟢 Bajo", "#27AE60"
        elif v <= RANGE_AMARILLO_MAX:
            return "🟡 Moderado", "#F4D03F"
        elif v <= RANGE_NARANJA_MAX:
            return "🟠 Alto", "#E67E22"
        else:
            return "🔴 Crítico", "#E74C3C"

    nivel_txt, nivel_color = clasificar_co2(valor_co2)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=valor_co2,
            number=dict(
                font=dict(size=FONT_GAUGE_NUMBER, color="#1a2e4a"),
                suffix=" grCO₂/kWh"
            ),
            gauge=dict(
                axis=dict(range=[GAUGE_MIN, GAUGE_MAX], tickfont=dict(size=FONT_GAUGE_TICK), tickcolor="#555"),
                bar=dict(color="#2C3E50", thickness=0.25),
                bgcolor="white",
                borderwidth=1,
                bordercolor="#CBD5E0",
                steps=[
                    dict(range=[GAUGE_MIN,          RANGE_VERDE_MAX],    color="#D5F5E3"),
                    dict(range=[RANGE_VERDE_MAX,    RANGE_AMARILLO_MAX], color="#FCF3CF"),
                    dict(range=[RANGE_AMARILLO_MAX, RANGE_NARANJA_MAX],  color="#FDEBD0"),
                    dict(range=[RANGE_NARANJA_MAX,  GAUGE_MAX],          color="#FADBD8"),
                ],
                threshold=dict(line=dict(color="red", width=3), thickness=0.75, value=RANGE_NARANJA_MAX)
            ),
            title=dict(
                text=f"<b>{fecha_sel} — {hora_sel:02d}:00h</b>",
                font=dict(size=FONT_SECTION_TITLE - 2, color="#1a2e4a")
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#F7F9FC", height=320,
            margin=dict(t=60, b=10, l=30, r=30),
            font=dict(family="Arial, sans-serif")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div style='background:#FFFFFF; border-radius:12px; padding:20px;
                    border-left: 6px solid {nivel_color}; margin-top:30px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <p style='font-size:{FONT_METRIC_LABEL}px; color:#666; margin:0;'>Nivel de alerta</p>
            <p style='font-size:{FONT_SECTION_TITLE}px; font-weight:700;
                      color:{nivel_color}; margin:4px 0;'>{nivel_txt}</p>
            <p style='font-size:{FONT_METRIC_LABEL + 2}px; color:#1a2e4a; margin:0;'>
                <b>{valor_co2:.1f}</b> grCO₂/kWh
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background:#FFFFFF; border-radius:12px; padding:20px;
                    border-left: 6px solid #3498DB; margin-top:30px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <p style='font-size:{FONT_METRIC_LABEL}px; color:#666; margin:0;'>Generación</p>
            <p style='font-size:{FONT_SECTION_TITLE}px; font-weight:700;
                      color:#2C3E50; margin:4px 0;'>{gen_mwh:,.1f}</p>
            <p style='font-size:{FONT_METRIC_LABEL}px; color:#666; margin:0;'>MWh</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='display:flex; gap:16px; margin-top:8px; flex-wrap:wrap;'>
        <span style='font-size:{FONT_ANNOTATION}px;'>🟢 Bajo: 0–{RANGE_VERDE_MAX} grCO₂/kWh</span>
        <span style='font-size:{FONT_ANNOTATION}px;'>🟡 Moderado: {RANGE_VERDE_MAX}–{RANGE_AMARILLO_MAX} grCO₂/kWh</span>
        <span style='font-size:{FONT_ANNOTATION}px;'>🟠 Alto: {RANGE_AMARILLO_MAX}–{RANGE_NARANJA_MAX} grCO₂/kWh</span>
        <span style='font-size:{FONT_ANNOTATION}px;'>🔴 Crítico: >{RANGE_NARANJA_MAX} grCO₂/kWh</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # ============================================================
    # NOTAS — PESTAÑA 1  ← AGREGA TUS NOTAS AQUÍ
    # ============================================================
    with st.expander("📝 Notas — Emisiones CO₂"):
        st.markdown("""
        - <!-- NOTA 1: escribe aquí tu primera nota sobre el medidor de emisiones -->
        - <!-- NOTA 2: escribe aquí observaciones sobre los rangos definidos -->
        - <!-- NOTA 3: agrega contexto metodológico o de fuente de datos si lo requieres -->
        """)
    # ============================================================


# ============================================================
# PESTAÑA 2 — PRECIO BOLSA vs PARTICIPACIÓN TÉRMICA
# ============================================================
with tab2:
    st.markdown("## 📊 Precio de Bolsa vs Participación Térmica")

    # Pestaña 2 no tiene filtros de usuario — el gráfico muestra toda la serie

    nino_start = pd.to_datetime(NINO_START)
    nino_end   = pd.to_datetime(NINO_END)

    # Agregar emisiones a nivel mensual desde df_emision (granularidad diaria → mensual)
    df_emision_mensual = (
        df_emision.copy()
        .assign(Fecha_mes=lambda d: pd.to_datetime(d["DateH"]).dt.to_period("M").dt.to_timestamp())
        .groupby("Fecha_mes", as_index=False)["grCO2/KWh"].mean()
        .rename(columns={"Fecha_mes": "Fecha", "grCO2/KWh": "emision_mensual"})
        .sort_values("Fecha")
    )

    fig_mercado = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>Precio de Bolsa (COP/kWh)</b>",
            "<b>Participación Térmica (%)</b>",
            "<b>Intensidad Promedio de Emisiones (grCO₂/kWh)</b>"
        ]
    )

    fig_mercado.add_trace(
        go.Scatter(
            x=df_grafico["Fecha"],
            y=df_grafico["valor"],
            mode="lines",
            name="Precio Bolsa (COP/kWh)",
            line=dict(color="#2980B9", width=2.5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Precio: %{y:,.1f} COP/kWh<extra></extra>"
        ),
        row=1, col=1
    )

    fig_mercado.add_trace(
        go.Scatter(
            x=df_grafico["Fecha"],
            y=df_grafico["participacion_pct"],
            mode="lines",
            name="Part. Térmica (%)",
            line=dict(color="#E74C3C", width=2.5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Participación: %{y:.1f}%<extra></extra>"
        ),
        row=2, col=1
    )

# Traza emisiones mensuales — fila 3
    fig_mercado.add_trace(
        go.Scatter(
            x=df_emision_mensual["Fecha"],
            y=df_emision_mensual["emision_mensual"],
            mode="lines",
            name="Emisiones (grCO₂/kWh)",
            line=dict(color="#27AE60", width=2.5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Emisiones: %{y:.1f} grCO₂/kWh<extra></extra>"
        ),
        row=3, col=1
    )

    for row_n in [1, 2, 3]:
        fig_mercado.add_vrect(
            x0=nino_start, x1=nino_end,
            fillcolor="#F39C12", opacity=0.15,
            layer="below", line_width=0,
            row=row_n, col=1
        )
        if row_n == 1:
            fig_mercado.add_annotation(
                x=nino_start + (nino_end - nino_start) / 2,
                y=1.15, yref="paper",
                text="<b>🌡️ Fenómeno del Niño<br>Dic 2023 – Abr 2024</b>",
                showarrow=False,
                font=dict(size=FONT_FENOMENO_NINO, color="#7D6608"),
                bgcolor="rgba(249,231,159,0.85)",
                bordercolor="#F4D03F",
                borderwidth=1, borderpad=4
            )
    fig_mercado.update_layout(
        paper_bgcolor="#F7F9FC", plot_bgcolor="#FFFFFF",
        height=720, hovermode="x unified",
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", font=dict(size=FONT_LEGEND)),
        margin=dict(t=80, b=50, l=60, r=30),
        font=dict(family="Arial, sans-serif")
    )
    fig_mercado.update_yaxes(title_text="COP/kWh",          title_font=dict(size=FONT_AXIS_TITLE), tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", row=1, col=1)
    fig_mercado.update_yaxes(title_text="Participación (%)", title_font=dict(size=FONT_AXIS_TITLE), tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", row=2, col=1)
    fig_mercado.update_yaxes(title_text="grCO₂/kWh",        title_font=dict(size=FONT_AXIS_TITLE), tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", row=3, col=1)
    fig_mercado.update_xaxes(title_text="Fecha",             title_font=dict(size=FONT_AXIS_TITLE), tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", showgrid=True, row=3, col=1)
    for ann in fig_mercado.layout.annotations:
        ann.font.size = FONT_SECTION_TITLE - 2
    st.plotly_chart(fig_mercado, use_container_width=True)

    st.markdown("---")
    # ============================================================
    # NOTAS — PESTAÑA 2  ← AGREGA TUS NOTAS AQUÍ
    # ============================================================
    with st.expander("📝 Notas — Precio Bolsa y Participación Térmica"):
        st.markdown("""
        - <!-- NOTA 1: escribe aquí observaciones sobre la relación precio bolsa / participación térmica -->
        - <!-- NOTA 2: puedes mencionar el impacto del Fenómeno del Niño en el período sombreado -->
        - <!-- NOTA 3: agrega referencias a fuentes de datos (XM, UPME, etc.) si aplica -->
        """)
    # ============================================================

# ============================================================
# PESTAÑA 2B — COMPARACIÓN DE MODELOS
# ============================================================
with tab2b:
    st.markdown("## 📊 Comparación de Modelos")

    @st.cache_data
    def cargar_comparacion():
        import os
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_modelos.csv")
        df = pd.read_csv(ruta)
        # Convertir columnas numéricas
        for c in ["val_rmse", "val_r2", "prod_rmse", "prod_r2", "dataset_size", "train_size", "val_size"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["fecha_ejecucion"] = pd.to_datetime(df["fecha_ejecucion"], errors="coerce")
        return df

    df_comp = cargar_comparacion()

    # ---- Tabla estilizada ----
    st.markdown(f"<p style='font-size:{FONT_METRIC_LABEL}px; color:#666;'>Resultados consolidados de todos los modelos evaluados. "
                "El mejor valor de cada métrica aparece resaltado en verde.</p>", unsafe_allow_html=True)

    # Columnas y etiquetas amigables
    COL_MAP = {
        "nombre_modelo":        "Modelo",
        "val_rmse":             "RMSE Validación",
        "val_r2":               "R² Validación",
        "prod_rmse":            "RMSE Producción",
        "prod_r2":              "R² Producción",
        "horizonte_prediccion": "Horizonte",
        "dataset_size":         "Dataset total",
        "train_size":           "Train",
        "val_size":             "Validación (n)",
        "fecha_ejecucion":      "Fecha ejecución",
    }
    df_show = df_comp.rename(columns=COL_MAP)

    # Formatear fecha
    if "Fecha ejecución" in df_show.columns:
        df_show["Fecha ejecución"] = df_show["Fecha ejecución"].dt.strftime("%Y-%m-%d %H:%M")

    # Formatear números
    for c in ["RMSE Validación", "RMSE Producción"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    for c in ["R² Validación", "R² Producción"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for c in ["Dataset total", "Train", "Validación (n)"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")

    # Identificar mejor modelo (menor RMSE producción)
    idx_mejor = df_comp["prod_rmse"].idxmin() if "prod_rmse" in df_comp.columns else None

    # Construir HTML de la tabla
    cols_tabla = [c for c in COL_MAP.values() if c in df_show.columns]
    header_html = "".join([
        f"<th style='background:#1a2e4a; color:#FFFFFF; padding:10px 14px; "
        f"font-size:{FONT_AXIS_TITLE - 2}px; font-weight:600; text-align:left; "  # ← tamaño encabezados tabla
        f"border-bottom:2px solid #2980B9; white-space:nowrap;'>{c}</th>"
        for c in cols_tabla
    ])

    filas_html = ""
    for i, row in df_show.iterrows():
        es_mejor = (i == idx_mejor)
        bg = "#E8F8F0" if es_mejor else ("#F7F9FC" if i % 2 == 0 else "#FFFFFF")
        borde = "border-left: 4px solid #27AE60;" if es_mejor else "border-left: 4px solid transparent;"
        celdas = "".join([
            f"<td style='padding:9px 14px; font-size:{FONT_AXIS_TICK}px; "  # ← tamaño celdas tabla
            f"color:#1a2e4a; border-bottom:1px solid #ECF0F1;'>{row[c]}</td>"
            for c in cols_tabla
        ])
        filas_html += f"<tr style='background:{bg}; {borde}'>{celdas}</tr>"

    tabla_html = f"""
    <div style='overflow-x:auto; border-radius:10px;
                box-shadow:0 2px 10px rgba(0,0,0,0.08); margin-top:12px;'>
        <table style='width:100%; border-collapse:collapse; font-family:Arial,sans-serif;'>
            <thead><tr>{header_html}</tr></thead>
            <tbody>{filas_html}</tbody>
        </table>
    </div>
    """
    st.markdown(tabla_html, unsafe_allow_html=True)

    # ---- Leyenda mejor modelo ----
    if idx_mejor is not None:
        nombre_mejor = df_comp.loc[idx_mejor, "nombre_modelo"]
        st.markdown(f"""
        <div style='margin-top:14px; padding:10px 16px; background:#E8F8F0;
                    border-left:5px solid #27AE60; border-radius:8px;
                    font-size:{FONT_METRIC_LABEL + 1}px; color:#1a2e4a;'>
            🏆 <b>Mejor modelo por RMSE en producción:</b> {nombre_mejor}
        </div>
        """, unsafe_allow_html=True)

    # ---- Gráfico de barras comparativo ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### Comparación visual de métricas")

    metric_opciones = {
        "RMSE Validación":   "val_rmse",
        "R² Validación":     "val_r2",
        "RMSE Producción":   "prod_rmse",
        "R² Producción":     "prod_r2",
    }
    metrica_sel = st.selectbox(
        "Métrica a comparar:",
        options=list(metric_opciones.keys()),
        index=2   # RMSE Producción por defecto
    )
    col_metrica = metric_opciones[metrica_sel]

    df_bar = df_comp[["nombre_modelo", col_metrica]].dropna().sort_values(col_metrica, ascending=True)

    colores = []
    for i in df_bar.index:
        colores.append("#27AE60" if i == idx_mejor else "#2980B9")

    fig_bar = go.Figure(go.Bar(
        x=df_bar[col_metrica],
        y=df_bar["nombre_modelo"],
        orientation="h",
        marker=dict(color=colores),
        text=df_bar[col_metrica].map(lambda v: f"{v:.4f}"),
        textposition="outside",
        textfont=dict(size=FONT_AXIS_TICK),   # ← tamaño etiquetas barras
        hovertemplate="<b>%{y}</b><br>" + metrica_sel + ": %{x:.4f}<extra></extra>"
    ))

    fig_bar.update_layout(
        title=dict(
            text=f"<b>{metrica_sel} por Modelo</b>",
            font=dict(size=FONT_SECTION_TITLE),   # ← tamaño título gráfico barras
            x=0.0
        ),
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#FFFFFF",
        height=max(280, len(df_bar) * 80),
        margin=dict(t=60, b=50, l=20, r=80),
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(
            title=metrica_sel,
            title_font=dict(size=FONT_AXIS_TITLE),   # ← tamaño título eje X
            tickfont=dict(size=FONT_AXIS_TICK),       # ← tamaño ticks eje X
            gridcolor="#ECF0F1", showgrid=True
        ),
        yaxis=dict(
            tickfont=dict(size=FONT_AXIS_TICK),       # ← tamaño etiquetas modelos
            automargin=True
        )
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    # ============================================================
    # NOTAS — PESTAÑA 2B  ← AGREGA TUS NOTAS AQUÍ
    # ============================================================
    with st.expander("📝 Notas — Comparación de Modelos"):
        st.markdown("""
        - <!-- NOTA 1: escribe aquí criterios de selección del mejor modelo -->
        - <!-- NOTA 2: comenta diferencias clave entre los modelos evaluados -->
        - <!-- NOTA 3: menciona condiciones del experimento (mismo dataset, mismo horizonte, etc.) -->
        """)
    # ============================================================

# ============================================================
# PESTAÑA 3 — RENDIMIENTO DEL MODELO
# ============================================================
with tab3:
    st.markdown("## 🤖 Rendimiento del Modelo de Predicción")

# ---- Filtros — solo visibles en esta pestaña ----
    import datetime

    modelos_disponibles = sorted(df_resultado["modelo"].unique())
    modelo_sel = st.selectbox("🤖 Modelo:", options=modelos_disponibles)

    # Rango de fechas disponibles en el modelo seleccionado
    fechas_mod    = pd.to_datetime(df_resultado[df_resultado["modelo"] == modelo_sel]["dateh"])
    fechas_mod_d  = pd.to_datetime(fechas_mod.dt.date.unique())

    años_mod   = sorted(fechas_mod_d.year.unique())
    meses_mod  = sorted(fechas_mod_d.month.unique())
    dias_mod   = sorted(fechas_mod_d.day.unique())

    MESES_ES = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril",
                5:"Mayo",  6:"Junio",   7:"Julio", 8:"Agosto",
                9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}

    # Valores predeterminados: fecha máxima disponible
    fecha_max     = fechas_mod_d.max()
    fecha_min_all = fechas_mod_d.min()

# Todos los índices apuntan a la fecha MÁXIMA (inicio = fin = fecha_max al abrir)
    idx_año_max = años_mod.index(fecha_max.year) if fecha_max.year in años_mod else len(años_mod) - 1

    # Meses y días del año/mes de la fecha máxima (iguales para inicio y fin)
    meses_max_disp = sorted(fechas_mod_d[fechas_mod_d.year == fecha_max.year].month.unique())
    idx_mes_max    = meses_max_disp.index(fecha_max.month) if fecha_max.month in meses_max_disp else len(meses_max_disp) - 1

    dias_max_disp  = sorted(fechas_mod_d[(fechas_mod_d.year == fecha_max.year) & (fechas_mod_d.month == fecha_max.month)].day.unique())
    idx_dia_max    = dias_max_disp.index(fecha_max.day) if fecha_max.day in dias_max_disp else len(dias_max_disp) - 1

    dias_min_disp  = sorted(fechas_mod_d[(fechas_mod_d.year == fecha_max.year) & (fechas_mod_d.month == fecha_max.month)].day.unique())
    idx_dia_min    = 0

    # Listas para selectores (inicio y fin usan la misma base: fecha_max)
    meses_mod     = meses_max_disp
    meses_mod_fin = meses_max_disp
    dias_mod_ini  = dias_min_disp
    dias_mod_fin  = dias_max_disp
    st.markdown("#### 📅 Rango de fechas")
    col_ai, col_mi, col_di, col_af, col_mf, col_df = st.columns(6)

    with col_ai:
        año_ini = st.selectbox("Año inicio:",  options=años_mod,      index=idx_año_max, key="año_ini")
    with col_af:
        año_fin = st.selectbox("Año fin:",     options=años_mod,      index=idx_año_max, key="año_fin")
    with col_mi:
        mes_ini = st.selectbox("Mes inicio:",  options=meses_mod,     index=idx_mes_max, key="mes_ini",
                               format_func=lambda m: MESES_ES[m])
    with col_mf:
        mes_fin = st.selectbox("Mes fin:",     options=meses_mod_fin, index=idx_mes_max, key="mes_fin",
                               format_func=lambda m: MESES_ES[m])
    with col_di:
        dia_ini = st.selectbox("Día inicio:",  options=dias_mod_ini,  index=idx_dia_min, key="dia_ini")
    with col_df:
        dia_fin = st.selectbox("Día fin:",     options=dias_mod_fin,  index=idx_dia_max, key="dia_fin")
    # Construir fechas de corte (con manejo de días inválidos en el mes)
    def fecha_segura(año, mes, dia):
        import calendar
        ultimo_dia = calendar.monthrange(año, mes)[1]
        dia_valido = min(dia, ultimo_dia)
        return datetime.datetime(año, mes, dia_valido)

    fecha_inicio = fecha_segura(año_ini, mes_ini, dia_ini)
    fecha_fin    = fecha_segura(año_fin, mes_fin, dia_fin)

    # Validación: si el usuario pone inicio > fin, avisar y no filtrar
    if fecha_inicio > fecha_fin:
        st.warning("⚠️ La fecha de inicio es mayor que la fecha de fin. Ajusta el rango.")
        fecha_inicio, fecha_fin = fecha_fin, fecha_inicio

    df_mod = df_resultado[
        (df_resultado["modelo"] == modelo_sel) &
        (df_resultado["dateh"] >= fecha_inicio) &
        (df_resultado["dateh"] <= fecha_fin)
    ].copy()

    if df_mod.empty:
        st.warning("No hay datos para el modelo y rango de fechas seleccionados.")

    if df_mod.empty:
        st.warning(f"No hay datos para el modelo: {modelo_sel}")
    else:
        mae  = df_mod["error_abs"].mean()
        rmse = np.sqrt((df_mod["error"] ** 2).mean())
        r2   = 1 - (
            np.sum((df_mod["valor_real"] - df_mod["prediccion"]) ** 2) /
            np.sum((df_mod["valor_real"] - df_mod["valor_real"].mean()) ** 2)
        )
        cobertura = (
            ((df_mod["valor_real"] >= df_mod["confianza_inferior"]) &
             (df_mod["valor_real"] <= df_mod["confianza_superior"])).mean() * 100
        ) if "confianza_inferior" in df_mod.columns else None

        mc1, mc2, mc3, mc4 = st.columns(4)
        cob_txt = f"{cobertura:.1f}%" if cobertura is not None and not np.isnan(cobertura) else "N/A"
        for col, label, val_txt, color in [
            (mc1, "MAE",          f"{mae:.3f}",  "#2980B9"),
            (mc2, "RMSE",         f"{rmse:.3f}", "#8E44AD"),
            (mc3, "R²",           f"{r2:.4f}",   "#27AE60"),
            (mc4, "Cobertura IC", cob_txt,       "#E67E22"),
        ]:
            col.markdown(f"""
            <div style='background:#FFFFFF; border-radius:10px; padding:16px;
                        border-top: 4px solid {color};
                        box-shadow: 0 2px 8px rgba(0,0,0,0.07); text-align:center;'>
                <p style='font-size:{FONT_METRIC_LABEL}px; color:#666; margin:0;'>{label}</p>
                <p style='font-size:{FONT_SECTION_TITLE}px; font-weight:700;
                          color:{color}; margin:4px 0;'>{val_txt}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Eje X compartido entre los dos gráficos via relayout en Plotly ──
        # Se usa un único xaxis vinculado: fig_mod con xaxis="x" y fig_err con xaxis="x"
        # Al hacer zoom/pan en cualquiera de los dos, el otro se sincroniza.

        fig_mod = go.Figure()

        if "confianza_inferior" in df_mod.columns and "confianza_superior" in df_mod.columns:
            fig_mod.add_trace(go.Scatter(
                x=pd.concat([df_mod["dateh"], df_mod["dateh"].iloc[::-1]]),
                y=pd.concat([df_mod["confianza_superior"], df_mod["confianza_inferior"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(52, 152, 219, 0.15)",
                line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
                name="Intervalo de confianza"
            ))

        fig_mod.add_trace(go.Scatter(
            x=pd.concat([df_mod["dateh"], df_mod["dateh"].iloc[::-1]]),
            y=pd.concat([
                df_mod["prediccion"] + df_mod["error_abs"],
                (df_mod["prediccion"] - df_mod["error_abs"]).iloc[::-1]
            ]),
            fill="toself", fillcolor="rgba(231, 76, 60, 0.05)",
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
            name="Banda de error absoluto"
        ))

        fig_mod.add_trace(go.Scatter(
            x=df_mod["dateh"], y=df_mod["valor_real"],
            mode="lines", name="Valor Real",
            line=dict(color="#1A252F", width=2),
            hovertemplate="<b>%{x}</b><br>Real: %{y:.4f}<extra></extra>"
        ))

        fig_mod.add_trace(go.Scatter(
            x=df_mod["dateh"], y=df_mod["prediccion"],
            mode="lines", name="Predicción",
            line=dict(color="#2980B9", width=2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Pred: %{y:.4f}<extra></extra>"
        ))

        fig_mod.update_layout(
            title=dict(text=f"<b>Real vs. Predicción — Modelo: {modelo_sel}</b>", font=dict(size=FONT_SECTION_TITLE), x=0.0),
            paper_bgcolor="#F7F9FC", plot_bgcolor="#FFFFFF", height=420, hovermode="x unified",
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=FONT_LEGEND)),
            margin=dict(t=60, b=70, l=60, r=30), font=dict(family="Arial, sans-serif"),
            # ← xaxis con matches vacío para que sea el "master" del zoom compartido
            xaxis=dict(
                title="Fecha/Hora", title_font=dict(size=FONT_AXIS_TITLE),
                tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", showgrid=True
            ),
            yaxis=dict(
                title="Valor", title_font=dict(size=FONT_AXIS_TITLE),
                tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", showgrid=True
            )
        )

        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=df_mod["dateh"], y=df_mod["error"],
            marker=dict(color=df_mod["error"].apply(lambda v: "rgba(231,76,60,0.7)" if v > 0 else "rgba(39,174,96,0.7)")),
            name="Error (Real − Predicción)",
            hovertemplate="<b>%{x}</b><br>Error: %{y:.4f}<extra></extra>"
        ))
        fig_err.add_hline(y=0, line_color="#1A252F", line_width=1.5)
        fig_err.update_layout(
            title=dict(text="<b>Error de Predicción en el Tiempo</b>", font=dict(size=FONT_SECTION_TITLE), x=0.0),
            paper_bgcolor="#F7F9FC", plot_bgcolor="#FFFFFF", height=300,
            margin=dict(t=55, b=50, l=60, r=30), font=dict(family="Arial, sans-serif"),
            # ← xaxis2 enlazado al xaxis de fig_mod via matches="x" en el HTML combinado
            xaxis=dict(
                title="Fecha/Hora", title_font=dict(size=FONT_AXIS_TITLE),
                tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1"
            ),
            yaxis=dict(
                title="Error", title_font=dict(size=FONT_AXIS_TITLE),
                tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", zeroline=False
            ),
            showlegend=False
        )
# ============================================================
        # CÁLCULO DE VENTANAS ROJA Y VERDE — Q1 y Q3 por día
        # Verde: horas donde predicción <= Q1 del día  ← umbral bajo diario
        # Roja:  horas donde predicción >= Q3 del día  ← umbral alto diario
        # ============================================================

        # Detectar paso temporal entre registros
        todos_ts = sorted(df_mod["dateh"].tolist())
        paso = (pd.Timestamp(todos_ts[1]) - pd.Timestamp(todos_ts[0])) if len(todos_ts) > 1 else pd.Timedelta(hours=1)

        # Calcular Q1 y Q3 de predicción por cada día calendario
        df_mod["_dia"] = df_mod["dateh"].dt.date
        cuartiles = df_mod.groupby("_dia")["prediccion"].quantile([0.25, 0.75]).unstack()
        cuartiles.columns = ["_q1", "_q3"]
        df_mod = df_mod.join(cuartiles, on="_dia")

        # Registros que cumplen cada condición (hora exacta)
        df_verde = df_mod[df_mod["prediccion"] <= df_mod["_q1"]].copy()
        df_rojo  = df_mod[df_mod["prediccion"] >= df_mod["_q3"]].copy()

        # Agrupar registros consecutivos en intervalos para vrect
        def construir_intervalos_horarios(df_filtrado, paso_temporal):
            if df_filtrado.empty:
                return []
            timestamps = sorted(df_filtrado["dateh"].tolist())
            intervalos = []
            inicio = pd.Timestamp(timestamps[0])
            fin    = pd.Timestamp(timestamps[0])
            for t in timestamps[1:]:
                t = pd.Timestamp(t)
                if (t - fin) <= paso_temporal * 1.5:
                    fin = t
                else:
                    intervalos.append((inicio, fin + paso_temporal))
                    inicio = t
                    fin    = t
            intervalos.append((inicio, fin + paso_temporal))
            return intervalos

        intervalos_rojo  = construir_intervalos_horarios(df_rojo,  paso)
        intervalos_verde = construir_intervalos_horarios(df_verde, paso)
        # ============================================================
        # ============================================================
        # ── Combinamos los dos en un solo figura con subplots para sincronizar el zoom ──
        fig_combined = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,   # ← ESTO es lo que sincroniza el zoom entre ambos
            vertical_spacing=0.12,
            subplot_titles=[
                f"<b>Real vs. Predicción — Modelo: {modelo_sel}</b>",
                "<b>Error de Predicción en el Tiempo</b>"
            ],
            row_heights=[0.65, 0.35]
        )

        # -- Copiar trazas de fig_mod --
# -- Copiar trazas de fig_mod --
        for trace in fig_mod.data:
            fig_combined.add_trace(trace, row=1, col=1)

        # -- Copiar trazas de fig_err --
        for trace in fig_err.data:
            fig_combined.add_trace(trace, row=2, col=1)

        # ── Sombreados ventana roja (ambas filas) ──────────────────
        for x0, x1 in intervalos_rojo:
            for row_n in [1, 2]:
                fig_combined.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor="rgba(180, 30, 20, 0.20)",   # ← color ventana roja
                    opacity=1, layer="below", line_width=0,
                    row=row_n, col=1
                )

        # ── Sombreados ventana verde (ambas filas) ─────────────────
        for x0, x1 in intervalos_verde:
            for row_n in [1, 2]:
                fig_combined.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor="rgba(20, 140, 60, 0.20)",   # ← color ventana verde
                    opacity=1, layer="below", line_width=0,
                    row=row_n, col=1
                )

        # ── Leyenda manual de ventanas en el título ────────────────
        fig_combined.add_annotation(
            x=1.0, y=1.06, xref="paper", yref="paper",
            text=("<b style='color:#E74C3C'>▌ Ventana Roja</b>: predicción ≥ Q3 del día"
                  "&nbsp;&nbsp;&nbsp;"
                  "<b style='color:#27AE60'>▌ Ventana Verde</b>: predicción ≤ Q1 del día"),
            showarrow=False,
            font=dict(size=FONT_ANNOTATION, color="#444"),   # ← tamaño leyenda ventanas
            xanchor="right", align="right",
            bgcolor="rgba(247,249,252,0.85)",
            bordercolor="#CBD5E0", borderwidth=1, borderpad=5
        )

        fig_combined.update_layout(
            paper_bgcolor="#F7F9FC", plot_bgcolor="#FFFFFF",
            height=750, hovermode="x unified",
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", font=dict(size=FONT_LEGEND)),
            margin=dict(t=80, b=60, l=60, r=30),
            font=dict(family="Arial, sans-serif")
        )
        fig_combined.update_yaxes(title_text="Valor", title_font=dict(size=FONT_AXIS_TITLE), tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", row=1, col=1)
        fig_combined.update_yaxes(title_text="Error", title_font=dict(size=FONT_AXIS_TITLE), tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", zeroline=False, row=2, col=1)
        fig_combined.update_xaxes(title_text="Fecha/Hora", title_font=dict(size=FONT_AXIS_TITLE), tickfont=dict(size=FONT_AXIS_TICK), gridcolor="#ECF0F1", showgrid=True, row=2, col=1)
        for ann in fig_combined.layout.annotations:
            ann.font.size = FONT_SECTION_TITLE - 2

        st.plotly_chart(fig_combined, use_container_width=True)

    st.markdown("---")
    # ============================================================
    # NOTAS — PESTAÑA 3  ← AGREGA TUS NOTAS AQUÍ
    # ============================================================
    with st.expander("📝 Notas — Rendimiento del Modelo"):
        st.markdown("""
        - <!-- NOTA 1: escribe aquí interpretación de las métricas MAE, RMSE, R² -->
        - <!-- NOTA 2: comenta sobre el comportamiento del error en períodos específicos -->
        - <!-- NOTA 3: menciona limitaciones del modelo o condiciones de validación -->
        """)
    # ============================================================


# ============================================================
# PESTAÑA 4 — METODOLOGÍA
# ============================================================
with tab4:
    st.markdown("## 📋 Metodología")
    st.markdown("---")

    # ============================================================
    # BLOQUE 1: INTRODUCCIÓN / CONTEXTO  ← ESCRIBE AQUÍ
    # ============================================================
    st.markdown("### 1. Contexto y Motivación")
    st.markdown("""
    El sistema eléctrico colombiano presenta una demanda altamente variable a lo largo del día, con picos que incrementan significativamente el despacho de generación térmica y, con ello, las emisiones de CO₂ asociadas a la matriz energética. Esta dinámica se agudiza en períodos de estrés hídrico como el Fenómeno del Niño, donde la menor disponibilidad de generación hidráulica obliga a una mayor participación de fuentes fósiles, elevando tanto el precio de bolsa como la intensidad de emisiones.
    En este contexto, contar con una herramienta que permita anticipar el comportamiento del precio de bolsa con horizonte de 24 horas representa un insumo valioso para la toma de decisiones operativas y de política energética. El presente trabajo busca contribuir a ese objetivo, facilitando la identificación de ventanas horarias donde sea posible redistribuir la demanda hacia períodos de menor intensidad de emisiones, promoviendo así una operación del sistema más limpia, eficiente y sostenible.
    """)
    st.markdown("---")

    # ============================================================
    # BLOQUE 2: FUENTES DE DATOS  ← ESCRIBE AQUÍ
    # ============================================================
    st.markdown("### 2. Fuentes de Datos")
    st.markdown("""
    Los datos utilizados en este trabajo provienen del sistema de información del mercado eléctrico colombiano y fueron procesados a partir de los siguientes archivos:

    aporte_hidricos_agrupados.parquet — Aportes hídricos agregados por período, utilizados como proxy de disponibilidad de generación hidráulica.
                
    demanda_comercial.parquet — Demanda comercial horaria del sistema, variable objetivo indirecta que condiciona el despacho de generación.
                
    generacion_completa.parquet — Registro detallado de generación por recurso y período, base para el análisis de la matriz energética.
                
    generacion_emision_agrupada.parquet — Generación agregada con su respectiva intensidad de emisiones en grCO₂/kWh, utilizada para el análisis ambiental del dashboard.
    
    intensidad_Co2.parquet — Serie histórica de intensidad de carbono del sistema, empleada como variable explicativa en los modelos de predicción.
    """)
    st.markdown("---")

    # ============================================================
    # BLOQUE 3: DESCRIPCIÓN DEL MODELO  ← ESCRIBE AQUÍ
    # ============================================================
    st.markdown("### 3. Modelo de Predicción")
    st.markdown("""
    <!-- Explica aquí:
         - Qué tipo de modelo(s) se usaron (ARIMA, XGBoost, LSTM, etc.)
         - Variable objetivo y variables explicativas
         - Período de entrenamiento y de prueba
         - Criterios de selección del modelo -->
    """)
    st.markdown("---")

    # ============================================================
    # BLOQUE 4: MÉTRICAS DE EVALUACIÓN  ← ESCRIBE AQUÍ
    # ============================================================
    st.markdown("### 4. Métricas de Evaluación")
    st.markdown("""
    La evaluación del desempeño de los modelos se realizó mediante cuatro métricas complementarias, seleccionadas para capturar tanto la magnitud del error como la calidad estadística de las predicciones:

    MAE — Error Absoluto Medio: Mide el promedio de las diferencias absolutas entre el valor real y la predicción. Es intuitivo e interpretable en las mismas unidades del precio de bolsa (COP/kWh), y es robusto frente a valores atípicos. Un MAE bajo indica que el modelo acierta consistentemente en magnitud.
    
    RMSE — Raíz del Error Cuadrático Medio: Penaliza de forma cuadrática los errores grandes, lo que lo hace especialmente sensible a los picos de precio, que son precisamente los momentos de mayor interés operativo. Se utilizó como métrica principal de selección del mejor modelo.
    
    R² — Coeficiente de Determinación: Indica la proporción de la varianza del precio de bolsa que es explicada por el modelo. Un R² cercano a 1 señala que el modelo captura adecuadamente la estructura temporal y los patrones de la serie. Permite comparar modelos en términos relativos independientemente de la escala de los datos.
    
    Cobertura del Intervalo de Confianza: Mide el porcentaje de observaciones reales que caen dentro del intervalo de confianza estimado por el modelo. Una cobertura cercana al 95% indica que la incertidumbre está correctamente cuantificada, lo cual es clave para la toma de decisiones bajo incertidumbre en el mercado eléctrico.
    """)
    st.markdown("---")

    # ============================================================
    # BLOQUE 5: LIMITACIONES Y TRABAJO FUTURO  ← ESCRIBE AQUÍ
    # ============================================================
    st.markdown("### 5. Limitaciones y Trabajo Futuro")
    st.markdown("""
    El modelo enfrenta limitaciones inherentes a la complejidad del mercado eléctrico colombiano. Entre los principales factores no incorporados se encuentran decisiones regulatorias y de política energética — como cambios en las reglas de despacho, contratos bilaterales o intervenciones del regulador — que alteran el comportamiento del precio de bolsa de forma abrupta y difícilmente predecible a partir de series históricas. De igual manera, variables climatológicas de mayor resolución espacial y temporal, como la precipitación regional o los índices de anomalía oceánica asociados a los fenómenos ENSO, podrían mejorar significativamente la capacidad predictiva del modelo en períodos de estrés hídrico.
                
    Como trabajo futuro se propone explorar la incorporación de estas variables exógenas mediante arquitecturas de modelos más flexibles, como redes neuronales recurrentes con atención o modelos híbridos que combinen componentes estructurales con aprendizaje automático. Adicionalmente, se plantea extender el horizonte de predicción más allá de las 24 horas y evaluar la transferibilidad del enfoque a otros mercados eléctricos de la región con características similares.
    """)
    st.markdown("---")


# ---- Pie de página ----
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:#95A5A6; font-size:{FONT_METRIC_LABEL}px;'>"
    "Dashboard Energético · Sector Eléctrico Colombiano · Análisis de Maestría</p>",
    unsafe_allow_html=True
)