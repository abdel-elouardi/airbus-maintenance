import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ══════════════════════════════════════════════
# STYLE INDUSTRIEL
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Airbus Maintenance",
    page_icon="✈️",
    layout="wide"
)

# CSS style industriel
st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .stMetric { background-color: #00205B; padding: 10px; border-radius: 8px; }
    h1, h2, h3 { color: #4FC3F7; }
    .stDataFrame { background-color: #001133; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CHARGER DONNÉES ET MODÈLES
# ══════════════════════════════════════════════
@st.cache_data
def load_data():
    cols = ['moteur', 'cycle', 'op1', 'op2', 'op3'] + [f'capteur{i}' for i in range(1,22)]
    df = pd.read_csv('data/raw/train_FD001.txt', sep='\s+', header=None, names=cols)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    cols_constantes = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=cols_constantes)
    cols_capteurs = [col for col in df.columns if 'capteur' in col]
    for col in cols_capteurs:
        if df[col].skew() > 1:
            df[col] = np.log1p(df[col] - df[col].min())
    corr = df[cols_capteurs].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    cols_corr = [c for c in upper.columns if any(upper[c] > 0.9)]
    df = df.drop(columns=cols_corr)
    max_cycles = df.groupby('moteur')['cycle'].max()
    df['cycle_max'] = df['moteur'].map(max_cycles)
    df['RUL'] = (df['cycle_max'] - df['cycle']).clip(upper=125)
    df = df.drop(columns=['cycle_max'])
    return df

@st.cache_resource
def load_models():
    return (
        joblib.load('models/model.pkl'),
        joblib.load('models/scaler.pkl'),
        joblib.load('models/model_fin_vie.pkl'),
        joblib.load('models/scaler_fin_vie.pkl'),
        joblib.load('models/model_prix.pkl'),
        joblib.load('models/scaler_prix.pkl'),
    )

df = load_data()
model_rul, scaler_rul, model_fv, scaler_fv, model_prix, scaler_prix = load_models()

# État de chaque moteur
cols_features = [col for col in df.columns if col not in ['moteur', 'cycle', 'RUL']]
etats = df.groupby('moteur').median().reset_index()
etats['RUL_predit']     = model_rul.predict(scaler_rul.transform(etats[cols_features]))
etats['decision']       = model_fv.predict(scaler_fv.transform(etats[cols_features]))
etats['prix']           = model_prix.predict(scaler_prix.transform(etats[cols_features]))
etats['decision_label'] = etats['decision'].map({0: "Recycler", 1: "Reparer", 2: "Vendre"})

# ══════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/24/Airbus_Logo_2017.png", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "📊 Vue Generale",
    "🔧 Maintenance",
    "♻️ Fin de Vie",
    "💰 Prix Revente",
    "📦 Stock"
])

# ══════════════════════════════════════════════
# PAGE 1 — VUE GENERALE
# ══════════════════════════════════════════════
if page == "📊 Vue Generale":
    st.title("✈️ Airbus — Systeme de Maintenance Predictive")
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✈️ Total Moteurs",    len(etats))
    c2.metric("🚨 Urgents (< 30)",   int((etats['RUL_predit'] < 30).sum()))
    c3.metric("⚠️ A surveiller",     int((etats['RUL_predit'].between(30,60)).sum()))
    c4.metric("✅ En bonne sante",   int((etats['RUL_predit'] >= 60).sum()))

    st.markdown("---")

    # Graphique RUL
    st.subheader("Distribution RUL des moteurs")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#001133')
    sns.histplot(etats['RUL_predit'], bins=20, color="#4FC3F7", ax=ax)
    ax.axvline(x=30, color='red',    linestyle='--', label='Urgent')
    ax.axvline(x=60, color='orange', linestyle='--', label='A surveiller')
    ax.tick_params(colors='white')
    ax.set_xlabel("Cycles restants", color='white')
    ax.legend()
    st.pyplot(fig)

# ══════════════════════════════════════════════
# PAGE 2 — MAINTENANCE
# ══════════════════════════════════════════════
elif page == "🔧 Maintenance":
    st.title("🔧 Maintenance Predictive")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dans 30 cycles", int((etats['RUL_predit'] <= 30).sum()))
    c2.metric("Dans 60 cycles", int((etats['RUL_predit'] <= 60).sum()))
    c3.metric("Dans 90 cycles", int((etats['RUL_predit'] <= 90).sum()))

    st.subheader("Moteurs a maintenir en priorite")
    urgents = etats[etats['RUL_predit'] < 60][['moteur', 'RUL_predit']].sort_values('RUL_predit')
    urgents.columns = ['Moteur', 'Cycles restants']
    urgents['Cycles restants'] = urgents['Cycles restants'].round(0)
    st.dataframe(urgents, use_container_width=True)

    st.subheader("Pieces les plus critiques")
    st.image("models/feature_importance.png")

# ══════════════════════════════════════════════
# PAGE 3 — FIN DE VIE
# ══════════════════════════════════════════════
elif page == "♻️ Fin de Vie":
    st.title("♻️ Classification Fin de Vie")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("♻️ Recycler", int((etats['decision'] == 0).sum()))
    c2.metric("🔧 Reparer",  int((etats['decision'] == 1).sum()))
    c3.metric("💰 Vendre",   int((etats['decision'] == 2).sum()))

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#001133')
    sns.countplot(x=etats['decision_label'],
                  hue=etats['decision_label'],
                  palette=["#FF5252", "#FF9800", "#4CAF50"],
                  ax=ax, legend=False)
    ax.tick_params(colors='white')
    ax.set_xlabel("Decision", color='white')
    ax.set_title("Distribution des decisions", color='white')
    st.pyplot(fig)

# ══════════════════════════════════════════════
# PAGE 4 — PRIX REVENTE
# ══════════════════════════════════════════════
elif page == "💰 Prix Revente":
    st.title("💰 Estimation Prix de Revente")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prix moyen", f"{etats['prix'].mean():.0f} EUR")
    c2.metric("Prix min",   f"{etats['prix'].min():.0f} EUR")
    c3.metric("Prix max",   f"{etats['prix'].max():.0f} EUR")

    ca = etats[etats['decision'] == 2]['prix'].sum()
    st.metric("💰 Chiffre d'affaire potentiel", f"{ca:,.0f} EUR")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#001133')
    sns.histplot(etats['prix'], bins=20, color="#4CAF50", ax=ax)
    ax.tick_params(colors='white')
    ax.set_xlabel("Prix (EUR)", color='white')
    st.pyplot(fig)

# ══════════════════════════════════════════════
# PAGE 5 — STOCK
# ══════════════════════════════════════════════
elif page == "📦 Stock":
    st.title("📦 Gestion du Stock")
    st.markdown("---")

    data = {
        "Horizon":    ["30 cycles", "60 cycles", "90 cycles"],
        "Moteurs":    [
            int((etats['RUL_predit'] <= 30).sum()),
            int((etats['RUL_predit'] <= 60).sum()),
            int((etats['RUL_predit'] <= 90).sum()),
        ]
    }
    df_stock = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#001133')
    sns.barplot(x="Horizon", y="Moteurs",
                hue="Horizon", data=df_stock,
                palette=["#4FC3F7", "#FF9800", "#FF5252"],
                ax=ax, legend=False)
    ax.tick_params(colors='white')
    ax.set_title("Maintenances prevues", color='white')
    st.pyplot(fig)

    st.dataframe(df_stock, use_container_width=True)