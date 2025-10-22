# app.py — Song Clustering with K-Means (Streamlit)

import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Song Clustering — K-Means", layout="wide")
st.title("Song Clustering — K-Means (Streamlit)")

# ---------- Helpers ----------
@st.cache_data
def load_csv(uploaded_or_path, is_file=True):
    if is_file:
        return pd.read_csv(uploaded_or_path)
    else:
        return pd.read_csv(uploaded_or_path)

def scale_df(df, features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    return pd.DataFrame(X, columns=features, index=df.index), scaler

def pca_2d(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X2d = pca.fit_transform(X)
    return pca, X2d

def elbow_silhouette_table(X, k_min, k_max, random_state=42):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, algorithm="elkan")
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(X, labels),
            "dbi": davies_bouldin_score(X, labels),
            "sse": km.inertia_,
        })
    return pd.DataFrame(rows)

def silhouette_per_cluster(X, labels):
    s = silhouette_samples(X, labels)
    return (pd.DataFrame({"cluster": labels, "sil": s})
            .groupby("cluster")["sil"].mean()
            .sort_values(ascending=False))

def cluster_profile(df, labels, features, label_col="cluster"):
    out = df.copy()
    out[label_col] = labels
    return out.groupby(label_col)[features].mean().round(3), out

# ---------- Sidebar controls ----------
st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader("Upload CSV (or use local path below)", type=["csv"])
local_path = st.sidebar.text_input("…or read local CSV path", value="single_genre_artists.csv")

use_defaults = st.sidebar.checkbox("Use built-in defaults for audio features", value=True)

default_id_cols = ['id_songs', 'name_song', 'name_artists', 'id_artists']
default_features = [
    'danceability','energy','loudness','speechiness',
    'acousticness','instrumentalness','liveness',
    'valence','tempo','duration_ms'
]

# Performance knobs
k_min, k_max = st.sidebar.slider("K sweep (elbow + silhouette)", 2, 15, (2, 8))
best_k = st.sidebar.number_input("Final K for K-Means", min_value=2, max_value=30, value=3, step=1)
sample_for_plot = st.sidebar.number_input("Max points to plot (PCA)", 2000, 50000, 20000, 1000)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42, 1)

# Optional speed-up: sample rows before everything (set 0 = no sampling)
row_sample = st.sidebar.number_input("OPTIONAL: sample rows before processing (0 = all)", 0, 200000, 0, 1000)

# ---------- Load data ----------
if uploaded is not None:
    df = load_csv(uploaded)
else:
    try:
        df = load_csv(local_path, is_file=False)
    except Exception as e:
        st.error(f"Could not read CSV. {e}")
        st.stop()

if row_sample and row_sample < len(df):
    df = df.sample(row_sample, random_state=random_state).reset_index(drop=True)

st.subheader("Preview")
st.dataframe(df.head(10), width='stretch')

with st.expander("Dataset info", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate rows", int(df.duplicated().sum()))
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

# ---------- Feature selection ----------
if use_defaults:
    for c in default_id_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    features = [f for f in default_features if f in df.columns]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = st.multiselect("Pick numeric features for clustering", options=numeric_cols,
                              default=[f for f in default_features if f in numeric_cols])

if len(features) < 2:
    st.warning("Pick at least 2 numeric features.")
    st.stop()

st.success(f"Using features: {features}")

# ---------- Scale ----------
X_scaled, scaler = scale_df(df, features)

with st.expander("Feature distributions (scaled)", expanded=False):
    ncols = 3
    rows = int(np.ceil(len(features)/ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(14, 4*rows))
    axes = axes.ravel()
    for i, col in enumerate(features):
        sns.histplot(X_scaled[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig, clear_figure=True)

# ---------- K sweep: elbow + silhouette ----------
st.subheader("Pick K (Elbow + Silhouette)")
with st.spinner("Computing elbow & silhouette across K…"):
    sweep = elbow_silhouette_table(X_scaled.values, k_min, k_max, random_state=random_state)

c1, c2 = st.columns(2)
with c1:
    st.line_chart(sweep.set_index("k")["sse"])
    st.caption("SSE (Elbow) — look for where the curve bends.")
with c2:
    st.line_chart(sweep.set_index("k")["silhouette"])
    st.caption("Silhouette — higher is better.")

st.dataframe(sweep.sort_values(["silhouette", "dbi"], ascending=[False, True]),
            width='stretch')

# ---------- Final K-Means ----------
st.subheader(f"Fit K-Means (k = {best_k})")
kmeans = KMeans(n_clusters=int(best_k), random_state=random_state, n_init=10, algorithm="elkan")
labels = kmeans.fit_predict(X_scaled.values)

sil = silhouette_score(X_scaled.values, labels)
dbi = davies_bouldin_score(X_scaled.values, labels)
colA, colB, colC = st.columns(3)
colA.metric("Silhouette (↑)", f"{sil:.3f}")
colB.metric("Davies–Bouldin (↓)", f"{dbi:.3f}")
colC.metric("Inertia / SSE", f"{kmeans.inertia_:,.0f}")

with st.expander("Silhouette by cluster", expanded=False):
    st.write(silhouette_per_cluster(X_scaled.values, labels))

# ---------- Cluster profiles ----------
prof, df_labeled = cluster_profile(df, labels, features, label_col="cluster")
st.subheader("Average feature values per cluster")
st.dataframe(prof, width='stretch')

# ---------- Visualizations ----------
st.subheader("Visualizations")

# PCA scatter with true K-Means centroids
pca, X2d = pca_2d(X_scaled.values, n_components=2, random_state=random_state)
plot_df = pd.DataFrame(X2d, columns=["PC1","PC2"])
plot_df["cluster"] = labels
if len(plot_df) > sample_for_plot:
    plot_df = plot_df.sample(sample_for_plot, random_state=random_state)
    
    
st.subheader("PCA with Centroids")
fig, ax = plt.subplots(figsize=(7,4))
sns.scatterplot(
    data=plot_df, x="PC1", y="PC2", hue="cluster",
    palette="tab10", s=18, alpha=0.7, edgecolor=None, ax=ax, legend=False
)
centroids_2d = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_2d[:,0], centroids_2d[:,1], marker="X", s=220, c="red", label="Centroids", zorder=10)
ax.set_title(f"PCA 2D — K-Means (k={best_k}) | Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
st.pyplot(fig, clear_figure=True)

# Bar chart of means
st.subheader("Bar chart of means")
fig, ax = plt.subplots(figsize=(10,6))
prof.T.plot(kind="bar", ax=ax)
ax.set_title("Cluster Profiles — Mean Feature Values")
ax.set_xlabel("Features"); ax.set_ylabel("Mean")
ax.legend(title="Cluster", bbox_to_anchor=(1.02,1), loc="upper left")
st.pyplot(fig, clear_figure=True)

# Heatmap
st.subheader("Heatmap for means")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(prof, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
ax.set_title("Cluster Profiles — Heatmap")
st.pyplot(fig, clear_figure=True)

# Boxplots for selected features
st.subheader("Distributions within clusters")

pick = st.multiselect(
    "Pick features to plot",
    options=features,
    default=[
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_ms'
    ]
)

cols = st.columns(3)  
for i, c in enumerate(pick):
    with cols[i % 3]:  
        fig, ax = plt.subplots(figsize=(4, 3))  
        sns.boxplot(
            x="cluster", y=c, data=df_labeled,
            palette="tab10", ax=ax, hue='cluster', legend=False
        )
        ax.set_title(f"{c} by Cluster", fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        st.pyplot(fig, clear_figure=True)

# ---------- Download ----------
st.subheader("Export")
csv_buf = io.StringIO()
df_labeled.to_csv(csv_buf, index=False)
st.download_button(
    "Download clustered dataset (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"clustered_songs_kmeans_k{best_k}.csv",
    mime="text/csv"
)

st.caption("Tip: Reduce K-sweep range or sample rows in the sidebar to speed up.")
