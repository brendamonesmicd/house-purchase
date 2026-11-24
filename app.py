"""App Streamlit: probabilidad de que un cliente compre una casa."""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
ANNUAL_RATE = 0.065  # tasa con la que el dataset calcula la mensualidad (EMI)
CAT = ["country", "city", "property_type", "furnishing_status"]

st.set_page_config(page_title="¿Comprará la casa?", page_icon="🏠", layout="centered")


@st.cache_resource
def load_assets():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    models = {s: joblib.load(ROOT / "models" / f"{s}.joblib") for s in meta["scenarios"]}
    return meta, models


def emi(loan, years, rate=ANNUAL_RATE):
    r, n = rate / 12, years * 12
    return loan * r / (1 - (1 + r) ** -n) if loan > 0 else 0.0


def prepare(df, scenario, meta):
    df = df.copy()
    df["loan_amount"] = (df["price"] - df["down_payment"]).clip(lower=0)
    df["emi_to_income_ratio"] = [round(emi(l, t) / s, 2) if s > 0 else 9.99
                                 for l, t, s in zip(df.loan_amount, df.loan_tenure_years, df.customer_salary)]
    feats = meta["scenarios"][scenario]["features"]
    X = df[feats].copy()
    if meta["scenarios"][scenario]["best_model"] == "lightgbm":
        for c in CAT:
            X[c] = pd.Categorical(X[c], categories=meta["categories"][c])
    return X, df


def predict(df, scenario, meta, models):
    X, full = prepare(df, scenario, meta)
    return models[scenario].predict_proba(X)[:, 1], full


meta, models = load_assets()

st.title("🏠 ¿Tu cliente comprará la casa?")
st.caption("Modelo entrenado con 200 000 operaciones del *Global House Purchase Decision Dataset* (Kaggle). "
           "Llena los datos que normalmente obtienes de la propiedad y del cliente.")

tab1, tab2, tab3 = st.tabs(["Un cliente", "Varios clientes (CSV)", "Sobre el modelo"])

with tab1:
    st.subheader("1. La propiedad")
    c1, c2 = st.columns(2)
    country = c1.selectbox("País", meta["categories"]["country"],
                           index=meta["categories"]["country"].index("USA"))
    city = c2.selectbox("Ciudad", meta["cities_by_country"][country])
    with st.form("cliente"):
        c1, c2 = st.columns(2)
        ptype = c1.selectbox("Tipo de propiedad", meta["categories"]["property_type"])
        furn = c2.selectbox("Amueblado", meta["categories"]["furnishing_status"])
        size = c1.number_input("Tamaño (pies²)", 300, 10000, 1800, step=50)
        price = c2.number_input("Precio de venta", 10_000, 10_000_000, 600_000, step=10_000)
        year = c1.number_input("Año de construcción", 1900, date.today().year, 2005)
        prev_owners = c2.number_input("Dueños anteriores", 0, 10, 1)
        rooms = c1.number_input("Recámaras", 1, 10, 3)
        baths = c2.number_input("Baños", 1, 10, 2)
        garage = c1.checkbox("Tiene cochera", True)
        garden = c2.checkbox("Tiene jardín", False)

        st.subheader("2. La zona")
        c1, c2 = st.columns(2)
        crime = c1.number_input("Delitos reportados en la zona (último año)", 0, 20, 1,
                                help="Dato público de seguridad de la colonia / vecindario.")
        legal = c2.selectbox("¿La propiedad tiene juicios o litigios?", ["No", "Sí"]) == "Sí"
        neigh = c1.slider("Calificación del vecindario", 1, 10, 6)
        conn = c2.slider("Conectividad (transporte, vías)", 1, 10, 6)

        st.subheader("3. El cliente")
        c1, c2 = st.columns(2)
        salary = c1.number_input("Ingreso mensual del cliente", 500, 500_000, 15_000, step=500)
        expenses = c2.number_input("Gastos mensuales del cliente", 0, 500_000, 6_000, step=500)
        down = c1.number_input("Enganche que puede dar", 0, 10_000_000, 150_000, step=5_000)
        tenure = c2.select_slider("Plazo del crédito (años)", [10, 15, 20, 25, 30], 20)

        visited = st.radio("¿El cliente ya visitó la propiedad?", ["Sí", "Todavía no"], horizontal=True)
        satisfaction = st.slider("¿Qué tanto le gustó? (1 = nada, 10 = le encantó)", 1, 10, 7,
                                 help="Pregúntale al cliente al terminar la visita. Se ignora si aún no la visita.")
        submitted = st.form_submit_button("Calcular probabilidad", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "country": country, "city": city, "property_type": ptype, "furnishing_status": furn,
            "property_size_sqft": size, "price": price, "constructed_year": year,
            "previous_owners": prev_owners, "rooms": rooms, "bathrooms": baths,
            "garage": int(garage), "garden": int(garden), "crime_cases_reported": crime,
            "legal_cases_on_property": int(legal), "customer_salary": salary,
            "loan_tenure_years": tenure, "monthly_expenses": expenses, "down_payment": down,
            "satisfaction_score": satisfaction, "neighbourhood_rating": neigh, "connectivity_score": conn,
        }])
        scenario = "con_visita" if visited == "Sí" else "sin_visita"
        p, full = predict(row, scenario, meta, models)
        p = float(p[0])
        ratio = float(full["emi_to_income_ratio"].iloc[0])
        loan = float(full["loan_amount"].iloc[0])

        st.divider()
        level = "alta" if p >= 0.6 else "media" if p >= 0.25 else "baja"
        color = {"alta": "green", "media": "orange", "baja": "red"}[level]
        st.markdown(f"### Probabilidad de compra: :{color}[{p:.0%}] ({level})")
        st.progress(min(max(p, 0.0), 1.0))
        base = meta["base_rate"]
        st.caption(f"Referencia: en el histórico compra el {base:.0%} de los clientes.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Crédito necesario", f"{loan:,.0f}")
        c2.metric("Mensualidad estimada", f"{emi(loan, tenure):,.0f}", help="Tasa anual de 6.5 %.")
        c3.metric("Mensualidad / ingreso", f"{ratio:.0%}")

        st.markdown("**Qué está pesando en la decisión**")
        checks = [
            (not legal, "La propiedad no tiene litigios", "La propiedad tiene litigios: casi nadie compra así"),
            (crime <= 2, "Zona con pocos delitos reportados (≤ 2)", f"{crime} delitos reportados: la zona ahuyenta compradores"),
            (ratio <= 0.39, "La mensualidad cabe en el ingreso (≤ 39 %)",
             f"La mensualidad es el {ratio:.0%} del ingreso: pide más enganche o más plazo"),
        ]
        if scenario == "con_visita":
            checks.insert(0, (satisfaction >= 7, "Al cliente le gustó la casa (≥ 7)",
                              "Al cliente no le convenció la casa (< 7)"))
        for ok, good, bad in checks:
            st.markdown(f"{'✅' if ok else '❌'} {good if ok else bad}")
        if scenario == "sin_visita":
            st.info("Sin visita el modelo no sabe si al cliente le gustará la casa, así que la probabilidad "
                    "se queda por debajo de ~40 % aun en el mejor caso. Agenda la visita y vuelve a calcular.")

with tab2:
    st.write("Sube un CSV con una fila por cliente y estas columnas (mismos nombres que el dataset):")
    cols = meta["scenarios"]["con_visita"]["features"]
    cols = [c for c in cols if c not in ("loan_amount", "emi_to_income_ratio")]
    st.code(",".join(cols), language=None)
    st.caption("`loan_amount` y `emi_to_income_ratio` se calculan solos. Si falta `satisfaction_score` "
               "se usa el modelo sin visita.")
    template = pd.DataFrame([dict(zip(cols, ["USA", "New York", "Apartment", "Unfurnished", 1500, 500000, 2010, 1, 3, 2,
                                             1, 0, 1, 0, 15000, 20, 6000, 150000, 8, 7, 7]))])
    st.download_button("Descargar plantilla", template.to_csv(index=False), "plantilla_clientes.csv", "text/csv")
    up = st.file_uploader("CSV de clientes", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            scenario = "con_visita" if "satisfaction_score" in df.columns else "sin_visita"
            need = [c for c in meta["scenarios"][scenario]["features"] if c not in ("loan_amount", "emi_to_income_ratio")]
            missing = [c for c in need if c not in df.columns]
            if missing:
                st.error(f"Faltan columnas: {', '.join(missing)}")
            else:
                p, full = predict(df, scenario, meta, models)
                out = df.assign(emi_to_income_ratio=full["emi_to_income_ratio"], prob_compra=np.round(p, 4))
                st.dataframe(out.sort_values("prob_compra", ascending=False), use_container_width=True)
                st.download_button("Descargar resultados", out.to_csv(index=False), "predicciones.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {e}")

with tab3:
    rows = []
    for s, info in meta["scenarios"].items():
        m = info["test_metrics"]
        rows.append({"Escenario": "Con visita" if s == "con_visita" else "Sin visita",
                     "Modelo": info["best_model"], "Accuracy": m["test/accuracy"], "ROC-AUC": m["test/roc_auc"],
                     "F1 (umbral 0.5)": m["test/f1"], "F1 (umbral ajustado)": m["test_tuned/f1"]})
    st.write("Métricas en el conjunto de **prueba** (20 %, 40 000 registros):")
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
    st.markdown("""
- División estratificada: **70 % entrenamiento, 10 % validación, 20 % prueba** (semilla 42).
- Se compararon regresión logística, árbol de decisión, random forest y LightGBM; se eligió el mejor por ROC-AUC en validación.
- La etiqueta del dataset sigue una regla casi exacta: compra si la satisfacción ≥ 7, no hay litigios,
  hay ≤ 2 delitos en la zona y la mensualidad es ≤ 39 % del ingreso. Por eso el modelo con visita acierta el 100 %.
  Con datos reales no esperes esa precisión.
""")
