import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# ══════════════════════════════════════════════
# 1. CHARGER ET NETTOYER
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
# 2. CALCULER LE RUL
# ══════════════════════════════════════════════
max_cycles = df.groupby('moteur')['cycle'].max()
df['cycle_max'] = df['moteur'].map(max_cycles)
df['RUL'] = (df['cycle_max'] - df['cycle']).clip(upper=125)
df = df.drop(columns=['cycle_max'])

# ══════════════════════════════════════════════
# 3. PRÉDIRE LE RUL POUR CHAQUE MOTEUR
# On prend le dernier cycle de chaque moteur
# = état actuel du moteur
# ══════════════════════════════════════════════
model_rul = joblib.load('models/model.pkl')
scaler    = joblib.load('models/scaler.pkl')

# Dernier état de chaque moteur

derniers_etats = df.groupby('moteur').median().reset_index()
cols_features  = [col for col in derniers_etats.columns 
                  if col not in ['moteur', 'cycle', 'RUL']]

X = scaler.transform(derniers_etats[cols_features])
derniers_etats['RUL_predit'] = model_rul.predict(X)
print(f"✅ RUL prédit pour {len(derniers_etats)} moteurs")

# ══════════════════════════════════════════════
# 4. GESTION DU STOCK
# On prédit combien de moteurs tombent en panne
# dans les 30, 60, 90 prochains cycles
# ══════════════════════════════════════════════
horizons = [30, 60, 90]
print("\n📊 Prévision des maintenances :")
for h in horizons:
    nb = (derniers_etats['RUL_predit'] <= h).sum()
    print(f"  Dans {h:3d} cycles : {nb:3d} moteurs à maintenir")

# ══════════════════════════════════════════════
# 5. CLASSIFICATION FIN DE VIE DU STOCK
# ══════════════════════════════════════════════
model_fin_vie = joblib.load('models/model_fin_vie.pkl')
scaler_fv     = joblib.load('models/scaler_fin_vie.pkl')

X_fv = scaler_fv.transform(derniers_etats[cols_features])
derniers_etats['decision'] = model_fin_vie.predict(X_fv)

labels = {0: "Recycler ♻️", 1: "Reparer 🔧", 2: "Vendre 💰"}
print("\n📊 Décisions pour les pièces :")
for k, v in labels.items():
    nb = (derniers_etats['decision'] == k).sum()
    print(f"  {v} : {nb} pieces")

# ══════════════════════════════════════════════
# 6. GRAPHIQUE
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution RUL prédit
sns.histplot(derniers_etats['RUL_predit'], bins=20,
             color="steelblue", ax=axes[0])
axes[0].set_title("Distribution RUL des moteurs")
axes[0].set_xlabel("RUL predit (cycles)")
axes[0].axvline(x=30, color='red', linestyle='--', label='Urgent')
axes[0].axvline(x=60, color='orange', linestyle='--', label='Bientot')
axes[0].legend()

# Distribution décisions
decisions = derniers_etats['decision'].map({0: "Recycler", 1: "Reparer", 2: "Vendre"})
sns.countplot(x=decisions, palette="coolwarm", ax=axes[1])
axes[1].set_title("Decisions fin de vie")

plt.tight_layout()
plt.savefig("models/gestion_stock.png")
print("\n✅ Graphique sauvegarde → models/gestion_stock.png")