# import pandas as pd
# import numpy as np
# import pickle

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# # =========================
# # 🔹 GLOBAL CONFIG
# # =========================
# RANDOM_STATE = 42
# TARGET_COLUMN = "calories_burned_kcal"


# # =========================
# # 🔹 LOAD & CLEAN DATA
# # =========================
# def load_and_clean_data(file_path):
#     """Load dataset and clean column names"""
#     df = pd.read_csv(file_path)

#     df.columns = (
#         df.columns
#         .str.strip()
#         .str.lower()
#         .str.replace(" ", "_")
#         .str.replace("(", "", regex=False)
#         .str.replace(")", "", regex=False)
#         .str.replace("/", "_")
#     )

#     # Fill missing numeric values
#     df = df.fillna(df.mean(numeric_only=True))

#     return df


# # =========================
# # 🔹 FEATURE ENGINEERING
# # =========================
# def create_features(df):
#     """Create additional features"""
#     df["intensity"] = df["avg_bpm"] * df["session_duration_hours"]
#     return df


# # =========================
# # 🔹 PREPROCESS DATA
# # =========================
# def preprocess_data(df):
#     """Encode and split features"""
#     df_encoded = pd.get_dummies(df, drop_first=True)

#     X = df_encoded.drop(TARGET_COLUMN, axis=1)
#     y = df_encoded[TARGET_COLUMN]

#     return X, y, df_encoded


# # =========================
# # 🔹 TRAIN REGRESSION MODEL
# # =========================
# def train_regression_model(X, y):
#     """Train and evaluate regression models"""

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
#     )

#     models = {
#         "Ridge": Ridge(),
#         "RandomForest": RandomForestRegressor(
#             n_estimators=100,
#             max_depth=10,
#             random_state=RANDOM_STATE
#         )
#     }

#     best_model = None
#     best_score = -1

#     print("\n📊 Regression Performance:\n")

#     for name, model in models.items():
#         # Cross-validation
#         cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
#         model.fit(X_train, y_train)

#         predictions = model.predict(X_test)

#         mae = mean_absolute_error(y_test, predictions)
#         rmse = np.sqrt(mean_squared_error(y_test, predictions))
#         r2 = r2_score(y_test, predictions)

#         print(f"{name}")
#         print(f"  CV R2: {cv_scores.mean():.3f}")
#         print(f"  MAE: {mae:.2f}")
#         print(f"  RMSE: {rmse:.2f}")
#         print(f"  R2: {r2:.3f}\n")

#         if r2 > best_score:
#             best_score = r2
#             best_model = model

#     return best_model, scaler, X.columns.tolist()


# # =========================
# # 🔹 CLUSTERING MODEL
# # =========================
# def train_clustering_model(df_encoded):
#     """Perform PCA + KMeans clustering"""

#     X_unsup = df_encoded.drop(TARGET_COLUMN, axis=1)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_unsup)

#     # PCA for dimensionality reduction
#     pca = PCA(n_components=5)
#     X_pca = pca.fit_transform(X_scaled)

#     # KMeans clustering
#     kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
#     clusters = kmeans.fit_predict(X_pca)

#     # Silhouette score
#     score = silhouette_score(X_pca, clusters)
#     print(f"\n🔥 Silhouette Score: {score:.3f}")

#     return kmeans, pca, scaler, clusters


# # =========================
# # 🔹 ANALYZE CLUSTERS
# # =========================
# def analyze_clusters(df, clusters):
#     """Print cluster insights"""

#     df["cluster"] = clusters

#     print("\n📊 Cluster Size Distribution:")
#     print(df["cluster"].value_counts())

#     numeric_cols = df.select_dtypes(include=np.number).columns

#     print("\n📊 Cluster Feature Means:")
#     print(df.groupby("cluster")[numeric_cols].mean())


# # =========================
# # 🔹 SAVE MODELS
# # =========================
# def save_models(model, scaler, columns, kmeans, pca, scaler_cluster):
#     """Save trained models"""
#     pickle.dump(model, open("model.pkl", "wb"))
#     pickle.dump(scaler, open("scaler.pkl", "wb"))
#     pickle.dump(columns, open("columns.pkl", "wb"))

#     pickle.dump(kmeans, open("kmeans.pkl", "wb"))
#     pickle.dump(pca, open("pca.pkl", "wb"))
#     pickle.dump(scaler_cluster, open("scaler_cluster.pkl", "wb"))

#     print("\n✅ All models saved successfully")


# # =========================
# # 🔹 MAIN FUNCTION
# # =========================
# def main():
#     print("🚀 Training Started...")

#     df = load_and_clean_data("Fitbit_dataset.csv")
#     df = create_features(df)

#     X, y, df_encoded = preprocess_data(df)

#     model, scaler, columns = train_regression_model(X, y)

#     kmeans, pca, scaler_cluster, clusters = train_clustering_model(df_encoded)

#     analyze_clusters(df, clusters)

#     save_models(model, scaler, columns, kmeans, pca, scaler_cluster)

#     print("\n⚡ DONE")


# # =========================
# # 🔹 ENTRY POINT
# # =========================
# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

# Clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# 🔹 CONFIG
# =========================
RANDOM_STATE = 42
TARGET = "calories_burned_kcal"


# =========================
# 🔹 LOAD DATA
# =========================
def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_")
    )

    return df


# =========================
# 🔹 HANDLE MISSING + OUTLIERS
# =========================
def clean_data(df):
    # Fill missing
    df = df.fillna(df.mean(numeric_only=True))

    # Outlier capping (IQR)
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.clip(df[col], lower, upper)

    return df


# =========================
# 🔹 FEATURE ENGINEERING
# =========================
def feature_engineering(df):
    df["intensity"] = df["avg_bpm"] * df["session_duration_hours"]
    return df


# =========================
# 🔹 PREPROCESS
# =========================
def preprocess(df):
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    return X, y, df


# =========================
# 🔹 TRAIN REGRESSION
# =========================
def train_regression(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
    }

    best_model = None
    best_r2 = -1
    best_metrics = {}

    print("\n📊 Regression Results:\n")

    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"{name}")
        print(f"  CV R2: {cv.mean():.3f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2: {r2:.3f}\n")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_metrics = {
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse
            }

    return best_model, scaler, X.columns.tolist(), best_metrics


# =========================
# 🔹 CLUSTERING
# =========================
def clustering(df_encoded):
    X_unsup = df_encoded.drop(TARGET, axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unsup)

    # PCA
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    score = silhouette_score(X_pca, clusters)
    print(f"\n🔥 Silhouette Score: {score:.3f}")

    return kmeans, pca, scaler, clusters


# =========================
# 🔹 ANALYZE CLUSTERS
# =========================
def analyze(df, clusters):
    df["cluster"] = clusters

    print("\n📊 Cluster Distribution:")
    print(df["cluster"].value_counts())

    print("\n📊 Cluster Means:")
    print(df.groupby("cluster").mean(numeric_only=True))


# =========================
# 🔹 SAVE
# =========================
def save_all(model, scaler, columns, kmeans, pca, scaler_cluster, metrics):
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(columns, open("columns.pkl", "wb"))

    pickle.dump(kmeans, open("kmeans.pkl", "wb"))
    pickle.dump(pca, open("pca.pkl", "wb"))
    pickle.dump(scaler_cluster, open("scaler_cluster.pkl", "wb"))

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    print("\n✅ All models saved")


# =========================
# 🔹 MAIN
# =========================
def main():
    print("🚀 Training Started...")

    df = load_data("Fitbit_dataset.csv")
    df = clean_data(df)
    df = feature_engineering(df)

    X, y, df_encoded = preprocess(df)

    model, scaler, columns, metrics = train_regression(X, y)

    kmeans, pca, scaler_cluster, clusters = clustering(df_encoded)

    analyze(df, clusters)

    save_all(model, scaler, columns, kmeans, pca, scaler_cluster, metrics)

    print("\n🎯 DONE")


# =========================
# 🔹 RUN
# =========================
if __name__ == "__main__":
    main()