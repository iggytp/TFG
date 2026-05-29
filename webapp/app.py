import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_FEATURES = [
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "e_Gmag",
    "e_BPmag",
    "e_RPmag",
]
MODEL_PATH = "modelo_clasificacion_estelar.joblib"
DATA_CANDIDATES = ["dataset_v1.csv", "dataset_predictoras_v1.csv"]
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def _safe_bounds(series: pd.Series) -> Tuple[float, float, float]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return 0.0, 1.0, 0.5

    q_low, q_high = clean.quantile([0.01, 0.99]).tolist()
    min_v = float(clean.min())
    max_v = float(clean.max())
    med_v = float(clean.median())

    low = min(min_v, q_low)
    high = max(max_v, q_high)
    if np.isclose(low, high):
        low = low - 1.0
        high = high + 1.0

    return float(low), float(high), med_v


@st.cache_resource
def load_model_artifact(model_path: str):
    model_file = PROJECT_ROOT / model_path
    if not model_file.exists():
        raise FileNotFoundError(
            f"No se encontro {model_file}. Ejecuta primero el notebook hasta generar el joblib."
        )

    artifact = joblib.load(model_file)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        features = artifact.get("features", DEFAULT_FEATURES)
        target = artifact.get("target", "clase_real")
        classes = artifact.get("classes", None)
        best_params = artifact.get("best_params", {})
        model_name = artifact.get("base_model_name", type(model).__name__)
    else:
        model = artifact
        features = DEFAULT_FEATURES
        target = "clase_real"
        classes = None
        best_params = {}
        model_name = type(model).__name__

    return model, features, target, classes, best_params, model_name, str(model_file)


@st.cache_data
def load_reference_dataset(features: List[str], target: str):
    csv_used = None
    for candidate in DATA_CANDIDATES:
        candidate_path = PROJECT_ROOT / candidate
        if candidate_path.exists():
            csv_used = candidate_path
            break

    if csv_used is None:
        return None, None, None

    df = pd.read_csv(csv_used)
    missing = [f for f in features if f not in df.columns]
    if missing:
        return str(csv_used), df, None

    stats = {}
    for feat in features:
        low, high, med = _safe_bounds(df[feat])
        stats[feat] = {
            "low": low,
            "high": high,
            "median": med,
        }

    class_dist = None
    if target in df.columns:
        class_dist = (
            df[target]
            .value_counts(normalize=True)
            .rename("ratio")
            .reset_index()
            .rename(columns={"index": target})
        )

    return str(csv_used), df, {"ranges": stats, "class_dist": class_dist}


def build_input_form(features: List[str], ranges: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    st.sidebar.header("Parametros de entrada")

    if "feature_values" not in st.session_state:
        st.session_state.feature_values = {
            feat: ranges.get(feat, {}).get("median", 0.0) for feat in features
        }

    if st.sidebar.button("Restaurar medianas", use_container_width=True):
        st.session_state.feature_values = {
            feat: ranges.get(feat, {}).get("median", 0.0) for feat in features
        }

    input_values = {}
    for feat in features:
        low = ranges.get(feat, {}).get("low", 0.0)
        high = ranges.get(feat, {}).get("high", 1.0)
        med = ranges.get(feat, {}).get("median", (low + high) / 2)

        current_value = st.session_state.feature_values.get(feat, med)
        current_value = float(np.clip(current_value, low, high))

        input_values[feat] = st.sidebar.slider(
            feat,
            min_value=float(low),
            max_value=float(high),
            value=float(current_value),
            step=float((high - low) / 300) if not np.isclose(high, low) else 0.01,
            format="%.6f",
        )

    st.session_state.feature_values = input_values
    return input_values


def predict_with_model(model, features: List[str], values: Dict[str, float]):
    X_input = pd.DataFrame([{feat: values[feat] for feat in features}])
    pred = model.predict(X_input)

    proba_df = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        classes = getattr(model, "classes_", [f"Clase_{i}" for i in range(len(probs))])
        proba_df = pd.DataFrame({
            "clase": [str(c) for c in classes],
            "probabilidad": probs,
        }).sort_values("probabilidad", ascending=False)

    return pred[0], X_input, proba_df


def sensitivity_curve(model, base_values: Dict[str, float], ranges, features: List[str], selected_feature: str):
    low = ranges[selected_feature]["low"]
    high = ranges[selected_feature]["high"]
    grid = np.linspace(low, high, 100)

    rows = []
    for v in grid:
        row = dict(base_values)
        row[selected_feature] = float(v)
        rows.append(row)

    X_grid = pd.DataFrame(rows)[features]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_grid)
        classes = [str(c) for c in getattr(model, "classes_", range(probs.shape[1]))]
        sens = pd.DataFrame({selected_feature: grid})
        for idx, cls in enumerate(classes):
            sens[f"P({cls})"] = probs[:, idx]
        sens_long = sens.melt(
            id_vars=[selected_feature],
            var_name="salida",
            value_name="valor",
        )
        return sens_long, selected_feature, "probabilidad"

    decision = model.decision_function(X_grid)
    sens = pd.DataFrame({selected_feature: grid, "score": decision})
    sens_long = sens.melt(id_vars=[selected_feature], var_name="salida", value_name="valor")
    return sens_long, selected_feature, "score"


def main():
    st.set_page_config(
        page_title="Clasificador Estelar Gaia DR3",
        page_icon="*",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Clasificador Estelar Interactivo")
    st.caption("Demo profesional para explorar predicciones del modelo entrenado en Gaia DR3.")

    try:
        model, features, target, classes, best_params, model_name, model_path_used = load_model_artifact(MODEL_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    csv_used, df_ref, ref = load_reference_dataset(features, target)
    ranges = ref["ranges"] if ref is not None else {f: {"low": 0.0, "high": 1.0, "median": 0.5} for f in features}

    col_a, col_b = st.columns([1.2, 1.2])
    with col_a:
        st.metric("Modelo", model_name)
    with col_b:
        st.metric("N variables", len(features))

    with st.expander("Ficha tecnica del modelo", expanded=False):
        st.write(f"Archivo cargado: {model_path_used}")
        st.write(f"Variables usadas: {features}")
        if classes is not None:
            st.write(f"Clases declaradas: {classes}")
        if best_params:
            st.write("Mejores hiperparametros guardados:")
            st.json(best_params)

    values = build_input_form(features, ranges)

    pred_class, X_input, proba_df = predict_with_model(model, features, values)

    st.subheader("Prediccion en tiempo real")
    top_row_1, top_row_2 = st.columns([1, 1])
    confidence = None
    with top_row_1:
        st.success(f"Clase predicha: {pred_class}")
    with top_row_2:
        if proba_df is not None:
            confidence = float(proba_df.iloc[0]["probabilidad"])
            st.info(f"Confianza estimada: {confidence:.2%}")
        else:
            st.info("Confianza no disponible para este estimador")

    st.subheader("Resumen rapido")
    st.caption("Interpretacion sencilla de la prediccion actual.")
    exec_col_a, exec_col_b = st.columns([1, 1])

    if proba_df is not None:
        top_probs = proba_df["probabilidad"].to_numpy()
        decision_margin = float(top_probs[0] - top_probs[1]) if len(top_probs) > 1 else float(top_probs[0])
    else:
        decision_margin = 0.0

    profile_alignment = 0.0
    if ref is not None:
        alignment_scores = []
        for feat in features:
            low = ranges[feat]["low"]
            high = ranges[feat]["high"]
            med = ranges[feat]["median"]
            span = max(high - low, 1e-9)
            distance_to_center = abs(values[feat] - med) / span
            alignment_scores.append(max(0.0, 1.0 - distance_to_center))
        profile_alignment = float(np.mean(alignment_scores)) if alignment_scores else 0.0

    with exec_col_a:
        st.metric("Ventaja frente a la 2a clase", f"{decision_margin:.2%}")
    with exec_col_b:
        st.metric("Perfil parecido al conjunto", f"{profile_alignment:.1%}")

    if confidence is not None:
        if confidence >= 0.8:
            st.success("Lectura clara: el modelo muestra una confianza alta en esta prediccion.")
        elif confidence >= 0.6:
            st.info("Lectura moderada: la prediccion es razonable, pero no totalmente contundente.")
        else:
            st.warning("Lectura cauta: la confianza es baja, conviene revisar o ajustar parametros.")

    if proba_df is not None:
        fig_probs = px.bar(
            proba_df,
            x="clase",
            y="probabilidad",
            color="probabilidad",
            color_continuous_scale="Teal",
            title="Distribucion de probabilidad por clase",
            text=proba_df["probabilidad"].map(lambda x: f"{x:.2%}"),
        )
        fig_probs.update_traces(textposition="outside")
        fig_probs.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(fig_probs, use_container_width=True)

    st.subheader("Comparativa de tus parametros frente al universo Gaia DR3")
    if df_ref is not None and ref is not None:
        comparison_rows = []
        for feat in features:
            series = df_ref[feat].dropna().astype(float)
            percentile = float((series < values[feat]).mean() * 100.0)
            comparison_rows.append(
                {
                    "feature": feat,
                    "valor_input": values[feat],
                    "percentil_aprox": percentile,
                    "mediana_referencia": ranges[feat]["median"],
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning("No hay datos de referencia para comparativas.")

    st.subheader("Analisis de sensibilidad (what-if)")
    selected_feature = st.selectbox("Variable a barrer", options=features, index=0)
    sens_long, x_feature, y_name = sensitivity_curve(model, values, ranges, features, selected_feature)

    fig_sens = px.line(
        sens_long,
        x=x_feature,
        y="valor",
        color="salida",
        title=f"Cambio en {y_name} al variar {selected_feature}",
    )
    if y_name == "probabilidad":
        fig_sens.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_sens, use_container_width=True)

    st.subheader("Vector de entrada actual")
    st.dataframe(X_input, use_container_width=True)


if __name__ == "__main__":
    main()
