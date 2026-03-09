# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dengue Early Warning System (SAT Dengue) — a Streamlit dashboard for visualizing and predicting dengue epidemic excess at the municipal level in Colombia. Academic project for MAIA (Universidad de los Andes).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The app runs at `http://localhost:8501`.

## Architecture

This is a single-file Streamlit application (`app.py`) with two data artifacts:

- **`logistic_regression.joblib`** — Serialized dict with keys `model` (LogisticRegression), `scaler` (StandardScaler), and `features` (list of 32 feature names). Loaded via `joblib.load()`.
- **`panel_municipal_mensual.csv`** — Municipal-level monthly panel dataset (~24MB). Key columns: `Departamento_Notificacion`, `Municipio_notificacion`, `ano`, `mes`, `casos_total`, `casos_regular`, `casos_grave`, `tasa_incidencia`, `temperatura_c`, `precipitacion_mm`, `ndvi`, `poblacion`.

### App Structure (app.py)

The app flows top-to-bottom in a single file with these sections:

1. **Config & styles** — Page config, CSS injection, `DEPT_COORDS` dict (department centroids for map)
2. **Data loading** — `@st.cache_resource` for model, `@st.cache_data` for CSV and predictions
3. **Prediction pipeline** — `generate_predictions()` applies the logistic regression model to compute `probabilidad_exceso` and `nivel_alerta` (Normal/Riesgo/Alerta based on thresholds 0.3/0.6)
4. **Sidebar filters** — Department, municipality, year, month selectors
5. **Row 1** — Scattergeo map (department-level alerts) + time series chart (historical vs predicted cases)
6. **Row 2** — Alert level probability chart + risk panel (summary stats with styled HTML)
7. **Export** — CSV download button for filtered period data
8. **Model info** — Expandable section with methodology details

### Key Thresholds

- **Normal**: probability < 0.3
- **Riesgo (Risk)**: 0.3 ≤ probability < 0.6
- **Alerta (Alert)**: probability ≥ 0.6
- Historical vs predicted split: year < 2024 is historical, ≥ 2024 is prediction

## Language

The UI, variable names, and comments are in **Spanish**. Maintain this convention.
