"""
Bogotá Property Price Predictor — Streamlit App
================================================
Loads cleaned_properati.csv, trains the Stacking Ensemble
(RF + XGBoost -> Ridge), and serves an interactive prediction UI.

Usage:
    streamlit run app.py
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, PolynomialFeatures, StandardScaler, TargetEncoder
)
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "cleaned_properati.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
RESIDENTIAL = ["Apartamento", "Casa", "Apartaestudio"]

ZONE_MAP = {
    "El Retiro": "Norte", "El Chicó": "Norte", "Los Rosales": "Norte",
    "Santa Ana": "Norte", "Usaquén": "Norte", "La Calleja": "Norte",
    "Cedritos": "Norte", "Bella Suiza": "Norte", "Barrancas": "Norte",
    "Chico Norte": "Norte", "Chico Norte Ii": "Norte", "San Patricio": "Norte",
    "Santa Barbara": "Norte", "El Virrey": "Norte", "Chico Reservado": "Norte",
    "Chico Navarra": "Norte", "Alhambra": "Norte", "Colina Campestre": "Norte",
    "Nueva Autopista": "Norte", "Prado Veraniego": "Norte", "Verbenal": "Norte",
    "La Uribe": "Norte", "El Contador": "Norte", "San Antonio Norte": "Norte",
    "La Macarena": "Norte",
    "Suba": "Noroccidental", "Cerros De Suba": "Noroccidental",
    "Colinas De Suba": "Noroccidental", "Suba Tibabuyes": "Noroccidental",
    "Pinar De Suba": "Noroccidental", "Britalia Norte": "Noroccidental",
    "Mazuren": "Noroccidental", "Portales Del Norte": "Noroccidental",
    "Villas De Granada": "Noroccidental", "Gran Granada": "Noroccidental",
    "Engativa": "Noroccidental", "Normandia": "Noroccidental",
    "Lagos De Cordoba": "Noroccidental", "Ciudadela Colsubsidio": "Noroccidental",
    "Santa Maria Del Lago": "Noroccidental", "Bosque De Pinos": "Noroccidental",
    "Metropolis": "Noroccidental", "Las Villas": "Noroccidental",
    "Los Lagartos": "Noroccidental", "Cofradía": "Noroccidental",
    "Chapinero": "Centro Norte", "Chapinero Alto": "Centro Norte",
    "Chapinero Central": "Centro Norte", "Barrios Unidos": "Centro Norte",
    "Teusaquillo": "Centro Norte", "La Soledad ": "Centro Norte",
    "Quinta Paredes": "Centro Norte", "Marly": "Centro Norte",
    "Palermo": "Centro Norte", "Galerias": "Centro Norte",
    "Niza": "Centro Norte", "Quinta Camacho": "Centro Norte",
    "El Batán": "Centro Norte", "Nicolas De Federman": "Centro Norte",
    "Lindaraja": "Centro Norte", "Cabrero": "Centro Norte",
    "San Diego": "Centro Norte", "Las Aguas": "Centro Norte",
    "Centro Internacional": "Centro Norte", "Parque Central Bavaria": "Centro Norte",
    "Alameda": "Centro Norte", "Florida Blanca": "Centro Norte",
    "Fontibón": "Occidente", "Modelia": "Occidente", "Villemar": "Occidente",
    "Hayuelos": "Occidente", "Ciudad Salitre": "Occidente",
    "El Salitre": "Occidente", "Puente Aranda": "Occidente",
    "Los Mártires": "Occidente", "Restrepo": "Occidente",
    "Antonio Nariño": "Occidente", "Roma": "Occidente",
    "Centenario": "Occidente", "Veraguas": "Occidente",
    "La Candelaria": "Centro", "Santa Fe": "Centro", "Las Nieves": "Centro",
    "La Sabana": "Centro", "Ricaurte": "Centro", "Pablo VI": "Centro",
    "Kennedy": "Sur", "Bosa": "Sur", "Bosa La Libertad": "Sur",
    "Ciudad Bolívar": "Sur", "Usme": "Sur", "Rafael Uribe Uribe": "Sur",
    "Tunjuelito": "Sur", "Madelena": "Sur", "Venecia": "Sur",
    "El Tunal": "Sur", "Patio Bonito": "Sur", "Tintala": "Sur",
    "Castilla": "Sur", "Nueva Castilla": "Sur", "Villa Alsacia": "Sur",
    "Gustavo Restrepo": "Sur", "Villas De Aranjuez": "Sur",
    "La Estancia": "Sur", "El Refugio": "Sur", "Santa Librada": "Sur",
    "San Cristobal": "Sur", "Quiroga": "Sur", "Turingia": "Sur",
    "La Salle": "Sur", "Ingles": "Sur",
}

RF_PARAMS = {
    "n_estimators": 200, "max_depth": 30, "min_samples_leaf": 5,
    "min_samples_split": 10, "max_features": 0.5,
    "random_state": 42, "n_jobs": -1,
}
XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.01,
    "subsample": 0.7, "colsample_bytree": 0.8,
    "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
}


# ── Data & model (cached so training only runs once per session) ──────────────
@st.cache_resource(show_spinner="Training model... (~30 s)")
def load_model():
    df = pd.read_csv(DATA_PATH)
    df = df[df["type"].isin(RESIDENTIAL)].reset_index(drop=True)

    location_counts = df["location"].value_counts()
    rare = location_counts[location_counts < 15].index
    df["location"] = df["location"].where(~df["location"].isin(rare), "Other")

    df["zone"]             = df["location"].map(ZONE_MAP).fillna("Otro")
    df["loc_count"]        = df["location"].map(df["location"].value_counts())
    df["bath_per_bed"]     = df["bathrooms"] / (df["bedrooms"] + 1)
    df["log_area_per_bed"] = df["log_area"]  - np.log(df["bedrooms"] + 1)
    df["total_rooms"]      = df["bedrooms"]  + df["bathrooms"]

    X = df.drop("log_price", axis=1)
    y = df["log_price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ]), ["bedrooms", "bathrooms", "log_area", "parking",
             "bath_per_bed", "log_area_per_bed", "total_rooms", "loc_count"]),
        ("te",  TargetEncoder(smooth="auto"), ["location", "zone"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["type"]),
    ])

    model = StackingRegressor(
        estimators=[
            ("rf",  Pipeline([("preprocessing", preprocessor),
                               ("model", RandomForestRegressor(**RF_PARAMS))])),
            ("xgb", Pipeline([("preprocessing", ColumnTransformer(transformers=[
                ("num", Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly",   PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ]), ["bedrooms", "bathrooms", "log_area", "parking",
                     "bath_per_bed", "log_area_per_bed", "total_rooms", "loc_count"]),
                ("te",  TargetEncoder(smooth="auto"), ["location", "zone"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["type"]),
            ])),
                               ("model", XGBRegressor(**XGB_PARAMS))])),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(np.exp(y_test), np.exp(y_pred)) / 1e6

    loc_counts = df["location"].value_counts()
    locations  = sorted(ZONE_MAP.keys())

    return model, loc_counts, locations, r2, rmse


def build_input(bedrooms, bathrooms, area_m2, parking, location, prop_type, loc_counts):
    location_enc = location if loc_counts.get(location, 0) >= 15 else "Other"
    log_area     = np.log(area_m2)
    return pd.DataFrame([{
        "location":         location_enc,
        "type":             prop_type,
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms,
        "parking":          int(parking),
        "log_area":         log_area,
        "zone":             ZONE_MAP.get(location, "Otro"),
        "loc_count":        loc_counts.get(location_enc, 1),
        "bath_per_bed":     bathrooms / (bedrooms + 1),
        "log_area_per_bed": log_area - np.log(bedrooms + 1),
        "total_rooms":      bedrooms + bathrooms,
    }])


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bogotá Property Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Bogotá Property Price Predictor")
st.caption("Stacking Ensemble · RF + XGBoost · Trained on Properati listings")

if not DATA_PATH.exists():
    st.error(f"Data file not found: `{DATA_PATH}`  \nRun `01_data_cleaning.ipynb` first.")
    st.stop()

model, loc_counts, locations, r2, rmse = load_model()

st.sidebar.header("Model performance")
st.sidebar.metric("R² (test set)", f"{r2:.3f}")
st.sidebar.metric("RMSE (test set)", f"{rmse:,.0f} M COP")
st.sidebar.caption("Trained once per session. Refresh the page to retrain.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    prop_type = st.selectbox("Property type", RESIDENTIAL)
    bedrooms  = st.number_input("Bedrooms",  min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    location  = st.selectbox("Neighborhood", locations)
    area_m2   = st.number_input("Area (m²)", min_value=10, max_value=2000, value=80)
    parking   = st.radio("Parking", ["Yes", "No"], horizontal=True)

zone    = ZONE_MAP.get(location, "Otro")
parking_int = 1 if parking == "Yes" else 0

st.caption(f"Zone: **{zone}**")

st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    row   = build_input(bedrooms, bathrooms, area_m2, parking_int, location, prop_type, loc_counts)
    price = float(np.exp(model.predict(row)[0]))

    st.success("Estimated price")
    c1, c2 = st.columns(2)
    c1.metric("COP", f"${price:,.0f}")
    c2.metric("Billions COP", f"{price / 1e9:.3f} B")

    st.caption(
        f"{bedrooms} bedrooms · {bathrooms} bathrooms · {area_m2} m² · "
        f"{'Parking' if parking_int else 'No parking'} · "
        f"{location} ({zone}) · {prop_type}"
    )
