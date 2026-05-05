import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
print(f"✅ Données chargées : {df.shape}")

# ══════════════════════════════════════════════
# 2. CALCULER LE RUL
# ══════════════════════════════════════════════
max_cycles = df.groupby('moteur')['cycle'].max()
df['cycle_max'] = df['moteur'].map(max_cycles)
df['RUL'] = (df['cycle_max'] - df['cycle']).clip(upper=125)
df = df.drop(columns=['cycle_max'])

# ══════════════════════════════════════════════
# 3. CRÉER LA DÉCISION
# Basée sur le RUL restant
# ══════════════════════════════════════════════
def decision(rul):
    if rul < 30:
        return 0  # Recycler ♻️
    elif rul < 80:
        return 1  # Réparer 🔧
    else:
        return 2  # Vendre 💰

df['decision'] = df['RUL'].apply(decision)
print(f"✅ Distribution des décisions :")
print(f"  Recycler : {(df['decision']==0).sum()}")
print(f"  Réparer  : {(df['decision']==1).sum()}")
print(f"  Vendre   : {(df['decision']==2).sum()}")

# ══════════════════════════════════════════════
# 4. SPLIT ET NORMALISER
# ══════════════════════════════════════════════
X = df.drop(columns=['moteur', 'cycle', 'RUL', 'decision'])
y = df['decision']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Train:{len(X_train)} Test:{len(X_test)}")

# ══════════════════════════════════════════════
# 5. ENTRAÎNER
# ══════════════════════════════════════════════
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")

# ══════════════════════════════════════════════
# 6. ÉVALUER
# ══════════════════════════════════════════════
y_pred = model.predict(X_test)
print("\n📊 Classification Report :")
print(classification_report(y_test, y_pred,
      target_names=["Recycler", "Reparer", "Vendre"]))

# ══════════════════════════════════════════════
# 7. GRAPHIQUE
# ══════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=["Recycler", "Reparer", "Vendre"],
            yticklabels=["Recycler", "Reparer", "Vendre"])
plt.title("Matrice de Confusion — Classification Fin de Vie")
plt.tight_layout()
plt.savefig("models/confusion_fin_vie.png")
print("✅ Graphique sauvegarde")

# ══════════════════════════════════════════════
# 8. SAUVEGARDER
# ══════════════════════════════════════════════
joblib.dump(model, 'models/model_fin_vie.pkl')
joblib.dump(scaler, 'models/scaler_fin_vie.pkl')
print("✅ Modele sauvegarde")