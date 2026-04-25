
import streamlit as st
import pickle
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import time

# =========================
# 🔹 CONFIG
# =========================
PAGE_TITLE = "🔥 Fitness AI"

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# 🔹 LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))

    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    pca = pickle.load(open("pca.pkl", "rb"))
    scaler_cluster = pickle.load(open("scaler_cluster.pkl", "rb"))

    return model, scaler, columns, kmeans, pca, scaler_cluster


# =========================
# 🔹 LOAD DATASET
# =========================
def load_dataset():
    df = pd.read_csv("Fitbit_dataset.csv")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_")
    )

    df["intensity"] = df["avg_bpm"] * df["session_duration_hours"]

    return df


# =========================
# 🔹 FEATURE GRAPH
# =========================
def show_feature_distributions(df):
    st.subheader("📊 Feature Distribution Overview")

    features = ["age", "weight_kg", "avg_bpm", "session_duration_hours"]

    cols = st.columns(2)

    for i, feature in enumerate(features):
        fig, ax = plt.subplots(figsize=(5, 3))

        ax.hist(df[feature], bins=20, color="steelblue", alpha=0.8)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")

        cols[i % 2].pyplot(fig)


# =========================
# 🔹 MODEL PERFORMANCE
# =========================
def show_model_performance(model, scaler, df):
    st.subheader("📈 Model Performance")

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("calories_burned_kcal", axis=1)
    y = df_encoded["calories_burned_kcal"]

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(y, preds, alpha=0.5, color="green")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")

    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    st.pyplot(fig)


# =========================
# 🔹 CLUSTER PLOT
# =========================
def show_cluster_plot(df, scaler_cluster, pca, kmeans):
    st.subheader("🔍 Workout Clusters")

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("calories_burned_kcal", axis=1)

    scaled = scaler_cluster.transform(X)
    reduced = pca.transform(scaled)
    clusters = kmeans.predict(reduced)

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.tab10.colors
    unique_clusters = np.unique(clusters)

    for i, c in enumerate(unique_clusters):
        data = reduced[clusters == c]

        ax.scatter(
            data[:, 0],
            data[:, 1],
            color=colors[i % 10],
            label=f"Cluster {c}",
            s=60,
            alpha=0.7
        )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Workout Clusters")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)


# =========================
# 🔹 MAIN APP
# =========================
def main():
    st.title(PAGE_TITLE)
    st.caption("AI Fitness Prediction System (Smooth Loading Version)")

    model, scaler, columns, kmeans, pca, scaler_cluster = load_models()
    df = load_dataset()

    # =========================
    # 🔥 INPUT SECTION
    # =========================
    st.subheader("👤 Predict Your Calories")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 10, 80, 25)
        weight = st.number_input("Weight (kg)", 30, 150, 70)

    with col2:
        bpm = st.number_input("Avg BPM", 60, 200, 100)
        duration = st.number_input("Duration (hrs)", 0.1, 5.0, 1.0)

    with col3:
        fat = st.number_input("Fat %", 0.0, 50.0, 20.0)
        freq = st.number_input("Workout Days", 0, 7, 3)

    # =========================
    # 🔥 PREDICT BUTTON
    # =========================
    if st.button("🚀 Predict", use_container_width=True):

        # =========================
        # ⚡ SPEED LOADING ANIMATION
        # =========================
        progress = st.progress(0)
        status = st.empty()

        for i in range(100):
            time.sleep(0.01)  # fast loading effect
            progress.progress(i + 1)
            status.text(f"Processing... {i+1}%")

        status.text("Finalizing results...")

        # =========================
        # 🔹 PREPARE DATA
        # =========================
        user = pd.DataFrame([{
            "age": age,
            "weight_kg": weight,
            "fat_percentage": fat,
            "workout_frequency_days_week": freq,
            "avg_bpm": bpm,
            "session_duration_hours": duration,
            "intensity": bpm * duration
        }])

        user = pd.get_dummies(user)

        for col in columns:
            if col not in user.columns:
                user[col] = 0

        user = user[columns]

        # =========================
        # 🔥 PREDICT
        # =========================
        calories = model.predict(scaler.transform(user))[0]
        cluster = kmeans.predict(pca.transform(scaler_cluster.transform(user)))[0]

        progress.empty()
        status.empty()

        # =========================
        # 🔥 RESULT (FIRST SHOW)
        # =========================
        st.success("✅ Prediction Completed!")

        st.subheader("📊 Result")

        st.metric("🔥 Calories Burned", f"{calories:.2f}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Base", f"{calories:.2f}")

        with col2:
            st.metric("After +50", f"{calories + 50:.2f}", "+50 kcal")

        with col3:
            st.metric("After -50", f"{calories - 50:.2f}", "-50 kcal")

        if calories < 200:
            st.info("Light Activity 🟢")
        elif calories < 400:
            st.warning("Moderate Activity 🟡")
        else:
            st.error("High Intensity 🔴")

        st.write(f"Cluster Group: {cluster}")

        # =========================
        # 📊 FEATURE GRAPH (AFTER RESULT)
        # =========================
        st.divider()
        show_feature_distributions(df)

        # =========================
        # 🔍 MORE INSIGHTS
        # =========================
        with st.expander("📈 Advanced Analysis"):
            show_model_performance(model, scaler, df)
            show_cluster_plot(df, scaler_cluster, pca, kmeans)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()