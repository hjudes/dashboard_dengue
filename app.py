"""
Sistema de Alerta Temprana de Dengue - Colombia
Dashboard interactivo — Basado en mockup Entrega 1
MAIA — Universidad de los Andes
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="SAT Dengue Colombia",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# COORDENADAS DE DEPARTAMENTOS (centroides aproximados)
# ============================================================
DEPT_COORDS = {
    'AMAZONAS': (-1.0, -71.9), 'ANTIOQUIA': (7.0, -75.5), 'ARAUCA': (7.0, -70.7),
    'ATLANTICO': (10.7, -75.0), 'BOGOTA': (4.6, -74.1), 'BOLIVAR': (8.7, -74.0),
    'BOYACA': (5.9, -73.4), 'CALDAS': (5.3, -75.5), 'CAQUETA': (1.5, -75.6),
    'CASANARE': (5.3, -71.3), 'CAUCA': (2.5, -76.8), 'CESAR': (9.3, -73.5),
    'CHOCO': (5.7, -76.7), 'CORDOBA': (8.3, -75.6), 'CUNDINAMARCA': (5.0, -74.0),
    'GUAINIA': (2.5, -69.0), 'GUAJIRA': (11.4, -72.4), 'GUAVIARE': (2.0, -72.6),
    'HUILA': (2.5, -75.7), 'MAGDALENA': (10.0, -74.0), 'META': (3.5, -73.0),
    'NARIÑO': (1.3, -78.0), 'NORTE SANTANDER': (7.9, -72.5), 'PUTUMAYO': (0.5, -76.0),
    'QUINDIO': (4.5, -75.7), 'RISARALDA': (5.0, -76.0), 'SAN ANDRES': (12.5, -81.7),
    'SANTANDER': (7.0, -73.2), 'SUCRE': (9.3, -75.4), 'TOLIMA': (4.0, -75.2),
    'VALLE': (3.8, -76.5), 'VAUPES': (1.0, -70.2), 'VICHADA': (4.5, -69.8)
}

# ============================================================
# ESTILOS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
    
    .stApp { font-family: 'DM Sans', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1b2838 0%, #1a3a5c 50%, #1a5276 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-bottom: 4px solid #e74c3c;
    }
    .main-header h1 {
        color: #ffffff;
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        margin: 0;
    }
    .main-header p {
        color: #a8b2d1;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
    }
    
    .risk-panel {
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    .risk-alta {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #dc2626;
    }
    .risk-moderada {
        background: linear-gradient(135deg, #fef9c3, #fef08a);
        border: 2px solid #ca8a04;
    }
    .risk-normal {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #16a34a;
    }
    
    .variable-row {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        font-size: 0.95rem;
    }
    .variable-row:last-child { border-bottom: none; }
    .variable-icon { font-size: 1.2rem; margin-right: 0.5rem; width: 28px; text-align: center; }
    .variable-label { color: #475569; flex: 1; }
    .variable-value { font-family: 'Space Mono', monospace; font-weight: 700; color: #1e293b; }
    
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE DATOS Y MODELO
# ============================================================
@st.cache_resource
def load_model():
    data = joblib.load("logistic_regression.joblib")
    return data['model'], data['scaler'], data['features']

@st.cache_data
def load_data():
    df = pd.read_csv("panel_municipal_mensual.csv")
    df['fecha'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str).str.zfill(2) + '-01')
    return df

@st.cache_data
def generate_predictions(_model, _scaler, features, df):
    df_pred = df.copy()
    mask = df_pred[features].notna().all(axis=1)
    df_valid = df_pred[mask]
    if len(df_valid) > 0:
        X = df_valid[features]
        X_scaled = _scaler.transform(X)
        probas = _model.predict_proba(X_scaled)[:, 1]
        df_pred.loc[mask, 'probabilidad_exceso'] = probas
    df_pred['nivel_alerta'] = pd.cut(
        df_pred['probabilidad_exceso'],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=['Normal', 'Riesgo', 'Alerta']
    )
    return df_pred

try:
    model, scaler, features = load_model()
    df = load_data()
    df = generate_predictions(model, scaler, features, df)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error cargando datos o modelo: {e}")

if not data_loaded:
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🦟 Sistema de Alerta Temprana de Dengue</h1>
    <p>Monitoreo y predicción de exceso epidémico a nivel municipal — Colombia</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Filtros de Consulta")
    st.markdown("---")
    
    meses_nombres = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    
    deptos = sorted(df['Departamento_Notificacion'].unique())
    depto_sel = st.selectbox("🗺️ Departamento", ["Todos"] + deptos)
    
    if depto_sel != "Todos":
        municipios = sorted(
            df[df['Departamento_Notificacion'] == depto_sel]['Municipio_notificacion'].unique()
        )
        muni_sel = st.selectbox("🏘️ Municipio", ["Todos"] + municipios)
    else:
        muni_sel = "Todos"
    
    st.markdown("---")
    
    anos_disponibles = sorted(df['ano'].unique(), reverse=True)
    ano_sel = st.selectbox("📅 Año", anos_disponibles, index=0)
    
    meses_disponibles = sorted(df[df['ano'] == ano_sel]['mes'].unique())
    mes_sel = st.selectbox(
        "🗓️ Mes", meses_disponibles,
        format_func=lambda x: meses_nombres.get(x, str(x)),
        index=len(meses_disponibles) - 1
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #94a3b8; font-size: 0.75rem;'>
        MAIA — Universidad de los Andes<br>
        Proyecto Desarrollo de Soluciones<br>
        2026
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DATOS FILTRADOS
# ============================================================
df_periodo = df[(df['ano'] == ano_sel) & (df['mes'] == mes_sel)].copy()
periodo_label = f"{meses_nombres[mes_sel]} {ano_sel}"

# ============================================================
# FILA 1: MAPA (izq) + SERIE TEMPORAL (der)
# ============================================================
col_map, col_ts = st.columns([1, 1])

with col_map:
    st.markdown('<div class="section-title">🗺️ Mapa de Alerta por Departamento</div>', unsafe_allow_html=True)
    
    df_map = (
        df_periodo[df_periodo['probabilidad_exceso'].notna()]
        .groupby('Departamento_Notificacion')
        .agg(
            prob_media=('probabilidad_exceso', 'mean'),
            casos=('casos_total', 'sum'),
            municipios_alerta=('probabilidad_exceso', lambda x: (x >= 0.5).sum()),
            total_municipios=('probabilidad_exceso', 'count')
        )
        .reset_index()
    )
    
    df_map['lat'] = df_map['Departamento_Notificacion'].map(lambda x: DEPT_COORDS.get(x, (4.5, -74.0))[0])
    df_map['lon'] = df_map['Departamento_Notificacion'].map(lambda x: DEPT_COORDS.get(x, (4.5, -74.0))[1])
    df_map['nivel'] = pd.cut(df_map['prob_media'], bins=[-0.01, 0.3, 0.6, 1.01], labels=['Normal', 'Riesgo', 'Alerta'])
    
    color_map = {'Normal': '#22c55e', 'Riesgo': '#eab308', 'Alerta': '#ef4444'}
    df_map['size'] = df_map['casos'].clip(lower=10) ** 0.35 * 4
    df_map['prob_pct'] = (df_map['prob_media'] * 100).round(1)
    
    fig_map = go.Figure()
    for nivel, color in color_map.items():
        df_nivel = df_map[df_map['nivel'] == nivel]
        if len(df_nivel) > 0:
            fig_map.add_trace(go.Scattergeo(
                lat=df_nivel['lat'], lon=df_nivel['lon'],
                text=df_nivel.apply(
                    lambda r: f"<b>{r['Departamento_Notificacion']}</b><br>"
                              f"Prob. exceso: {r['prob_pct']}%<br>"
                              f"Casos: {int(r['casos'])}<br>"
                              f"Mun. alerta: {int(r['municipios_alerta'])}/{int(r['total_municipios'])}",
                    axis=1),
                hoverinfo='text',
                marker=dict(size=df_nivel['size'], color=color, opacity=0.8,
                           line=dict(width=1, color='white'), sizemin=8),
                name=nivel
            ))
    
    fig_map.update_geos(
        scope='south america', center=dict(lat=4.5, lon=-74.0),
        projection_scale=5.5,
        showland=True, landcolor='#f1f5f9',
        showocean=True, oceancolor='#e0f2fe',
        showcountries=True, countrycolor='#94a3b8',
        showframe=False, bgcolor='rgba(0,0,0,0)'
    )
    fig_map.update_layout(
        height=480, margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5, font=dict(size=11)),
        font_family="DM Sans"
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_ts:
    st.markdown('<div class="section-title">📈 Predicción de Casos de Dengue</div>', unsafe_allow_html=True)
    
    if muni_sel != "Todos":
        df_ts = df[df['Municipio_notificacion'] == muni_sel].sort_values('fecha')
    elif depto_sel != "Todos":
        df_ts = df[df['Departamento_Notificacion'] == depto_sel].groupby('fecha').agg(
            casos_total=('casos_total', 'sum'), casos_regular=('casos_regular', 'sum'),
            casos_grave=('casos_grave', 'sum'), probabilidad_exceso=('probabilidad_exceso', 'mean')
        ).reset_index()
    else:
        df_ts = df.groupby('fecha').agg(
            casos_total=('casos_total', 'sum'), casos_regular=('casos_regular', 'sum'),
            casos_grave=('casos_grave', 'sum'), probabilidad_exceso=('probabilidad_exceso', 'mean')
        ).reset_index()
    
    df_ts_hist = df_ts[df_ts['fecha'].dt.year < 2024]
    df_ts_pred = df_ts[df_ts['fecha'].dt.year >= 2024]
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts_hist['fecha'], y=df_ts_hist['casos_total'],
        name='Casos históricos', line=dict(color='#3b82f6', width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.1)', mode='lines'
    ))
    if len(df_ts_pred) > 0:
        fig_ts.add_trace(go.Scatter(
            x=df_ts_pred['fecha'], y=df_ts_pred['casos_total'],
            name='Predicción', line=dict(color='#ef4444', width=2.5, dash='dot'),
            fill='tozeroy', fillcolor='rgba(239,68,68,0.08)', mode='lines'
        ))
    fig_ts.update_layout(
        height=450, font_family="DM Sans",
        plot_bgcolor='#fafafa', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=40, l=50, r=20), yaxis_title="Casos",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified'
    )
    fig_ts.update_xaxes(gridcolor='#e5e7eb')
    fig_ts.update_yaxes(gridcolor='#e5e7eb')
    st.plotly_chart(fig_ts, use_container_width=True)

# ============================================================
# FILA 2: NIVEL DE ALERTA (izq) + PANEL DE RIESGO (der)
# ============================================================
col_alert, col_risk = st.columns([1, 1])

with col_alert:
    st.markdown('<div class="section-title">⚠️ Nivel de Alerta</div>', unsafe_allow_html=True)
    
    if muni_sel != "Todos":
        df_alert = df[df['Municipio_notificacion'] == muni_sel].sort_values('fecha')
    elif depto_sel != "Todos":
        df_alert = df[df['Departamento_Notificacion'] == depto_sel].groupby('fecha').agg(
            probabilidad_exceso=('probabilidad_exceso', 'mean'), casos_total=('casos_total', 'sum')
        ).reset_index()
    else:
        df_alert = df.groupby('fecha').agg(
            probabilidad_exceso=('probabilidad_exceso', 'mean'), casos_total=('casos_total', 'sum')
        ).reset_index()
    
    df_alert = df_alert.dropna(subset=['probabilidad_exceso'])
    
    fig_alert = go.Figure()
    fig_alert.add_hrect(y0=0, y1=0.3, fillcolor="rgba(34,197,94,0.1)", line_width=0)
    fig_alert.add_hrect(y0=0.3, y1=0.6, fillcolor="rgba(234,179,8,0.1)", line_width=0)
    fig_alert.add_hrect(y0=0.6, y1=1.0, fillcolor="rgba(239,68,68,0.1)", line_width=0)
    
    fig_alert.add_trace(go.Scatter(
        x=df_alert['fecha'], y=df_alert['probabilidad_exceso'],
        name='Prob. Exceso', line=dict(color='#1e293b', width=2.5),
        mode='lines', fill='tozeroy', fillcolor='rgba(30,41,59,0.08)'
    ))
    fig_alert.add_hline(y=0.3, line_dash="dash", line_color="#22c55e", line_width=1,
                        annotation_text="Normal", annotation_position="right",
                        annotation_font_color="#22c55e", annotation_font_size=10)
    fig_alert.add_hline(y=0.6, line_dash="dash", line_color="#eab308", line_width=1,
                        annotation_text="Riesgo", annotation_position="right",
                        annotation_font_color="#eab308", annotation_font_size=10)
    
    fig_alert.update_layout(
        height=380, font_family="DM Sans",
        plot_bgcolor='#fafafa', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=40, l=50, r=60),
        yaxis=dict(title="Probabilidad", range=[0, 1], gridcolor='#e5e7eb'),
        xaxis=dict(gridcolor='#e5e7eb'),
        showlegend=False, hovermode='x unified'
    )
    st.plotly_chart(fig_alert, use_container_width=True)

with col_risk:
    st.markdown('<div class="section-title">🎯 Panel de Riesgo</div>', unsafe_allow_html=True)
    
    if muni_sel != "Todos":
        df_risk = df_periodo[df_periodo['Municipio_notificacion'] == muni_sel]
        label_risk = muni_sel
    elif depto_sel != "Todos":
        df_risk = df_periodo[df_periodo['Departamento_Notificacion'] == depto_sel]
        label_risk = depto_sel
    else:
        df_risk = df_periodo
        label_risk = "Nacional"
    
    if len(df_risk) > 0:
        prob_media = df_risk['probabilidad_exceso'].mean()
        casos_total = int(df_risk['casos_total'].sum())
        temp_media = df_risk['temperatura_c'].mean()
        precip_media = df_risk['precipitacion_mm'].mean()
        pob_total = df_risk['poblacion'].sum()
        ndvi_media = df_risk['ndvi'].mean()
        
        if pd.notna(prob_media):
            if prob_media >= 0.6:
                nivel_text, panel_class, nivel_emoji, prob_color = "ALTO", "risk-alta", "🔴", "#dc2626"
            elif prob_media >= 0.3:
                nivel_text, panel_class, nivel_emoji, prob_color = "MODERADO", "risk-moderada", "🟡", "#ca8a04"
            else:
                nivel_text, panel_class, nivel_emoji, prob_color = "NORMAL", "risk-normal", "🟢", "#16a34a"
        else:
            nivel_text, panel_class, nivel_emoji, prob_color = "N/D", "risk-normal", "⚪", "#94a3b8"
            prob_media = 0
        
        st.markdown(f"""
        <div class="{panel_class}" style="margin-bottom: 1rem; border-radius: 12px; padding: 1.5rem;">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">⚠️</span>
                <span style="font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.1rem;">
                    Probabilidad de brote: <span style="color: {prob_color};">{prob_media*100:.0f}%</span>
                </span>
            </div>
            <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem;">
                Nivel: {nivel_emoji} {nivel_text}
            </div>
            <div style="font-size: 0.8rem; color: #475569;">
                {label_risk} — {periodo_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: #ffffff; border-radius: 12px; padding: 1.2rem 1.5rem; 
                    box-shadow: 0 2px 12px rgba(0,0,0,0.06);">
            <div style="font-weight: 700; font-size: 0.9rem; margin-bottom: 0.8rem; color: #1e293b;">
                Variables
            </div>
            <div class="variable-row">
                <span class="variable-icon">🌧️</span>
                <span class="variable-label">Lluvia</span>
                <span class="variable-value">{precip_media:.0f} mm</span>
            </div>
            <div class="variable-row">
                <span class="variable-icon">🌡️</span>
                <span class="variable-label">Temperatura</span>
                <span class="variable-value">{temp_media:.1f}°C</span>
            </div>
            <div class="variable-row">
                <span class="variable-icon">🌿</span>
                <span class="variable-label">NDVI (Vegetación)</span>
                <span class="variable-value">{ndvi_media:.3f}</span>
            </div>
            <div class="variable-row">
                <span class="variable-icon">👥</span>
                <span class="variable-label">Población</span>
                <span class="variable-value">{pob_total:,.0f}</span>
            </div>
            <div class="variable-row">
                <span class="variable-icon">🏥</span>
                <span class="variable-label">Casos en el mes</span>
                <span class="variable-value">{casos_total:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No hay datos para el período y ubicación seleccionados.")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# BOTÓN DESCARGAR REPORTE
# ============================================================
st.markdown("---")
col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 2])
with col_dl2:
    if len(df_periodo) > 0:
        df_export = (
            df_periodo[df_periodo['probabilidad_exceso'].notna()]
            [['Departamento_Notificacion', 'Municipio_notificacion', 'ano', 'mes',
              'casos_total', 'tasa_incidencia', 'temperatura_c', 'precipitacion_mm',
              'probabilidad_exceso', 'nivel_alerta']]
            .sort_values('probabilidad_exceso', ascending=False).copy()
        )
        df_export.columns = ['Departamento', 'Municipio', 'Año', 'Mes', 'Casos',
                             'Tasa Incidencia', 'Temperatura (°C)', 'Precipitación (mm)',
                             'Probabilidad Exceso', 'Nivel Alerta']
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Descargar Reporte",
            data=csv_buffer.getvalue(),
            file_name=f"reporte_alerta_dengue_{ano_sel}_{mes_sel:02d}.csv",
            mime="text/csv", use_container_width=True
        )

# ============================================================
# INFO DEL MODELO
# ============================================================
with st.expander("ℹ️ Acerca del Modelo y la Metodología"):
    st.markdown("""
    **Modelo:** Regresión Logística con regularización L2 y ponderación de clases balanceada.
    
    **Métricas de evaluación (test — año 2024):**
    - ROC-AUC: **0.9156** | F1-Score (exceso): **0.5774** | Recall (exceso): **87.7%**
    
    **Features utilizados (32):**
    Variables climáticas actuales y con rezagos de 1-3 meses (temperatura, precipitación, NDVI, punto de rocío),
    casos y tasas de incidencia con rezagos, medias móviles de 3 meses, proporciones epidemiológicas y población.
    
    **Definición de exceso:** Un municipio presenta exceso epidémico cuando los casos superan
    el umbral histórico (media + 2 desviaciones estándar).
    
    **Fuentes de datos:**
    - Epidemiológicos: SIVIGILA (2010, 2016, 2019, 2022, 2024)
    - Climáticos: Google Earth Engine (ERA5-Land, CHIRPS, MODIS)
    - Demográficos: DANE — Proyecciones de población municipal
    
    **Nota:** Prototipo académico. No reemplaza protocolos oficiales de vigilancia epidemiológica.
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.75rem; margin-top:2rem; padding:1rem;">
    Sistema de Alerta Temprana de Dengue — Microproyecto PDS<br>
    MAIA — Universidad de los Andes — 2026<br>
    Danilo Camargo · Jhon Eduard Moreno Díaz · Hernán Javier Silva Sosa · Sheyla Ruby Zela Quirita
</div>
""", unsafe_allow_html=True)
