import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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
# 3. CRÉER LE PRIX
# Prix simulé basé sur le RUL
# Plus le RUL est élevé → plus la pièce vaut cher
# Prix de base : 10 000€ pour une pièce neuve
# ══════════════════════════════════════════════
PRIX_NEUF = 10000  # prix d'une pièce neuve en €
df['prix'] = (df['RUL'] / 125) * PRIX_NEUF
# Ajouter un peu de bruit pour simuler le marché
np.random.seed(42)
df['prix'] = df['prix'] * (1 + np.random.normal(0, 0.1, len(df)))
df['prix'] = df['prix'].clip(lower=500)  # prix minimum 500€
print(f"✅ Prix simulé — Min:{df['prix'].min():.0f}€ Max:{df['prix'].max():.0f}€ Moyenne:{df['prix'].mean():.0f}€")

# ══════════════════════════════════════════════
# 4. SPLIT ET NORMALISER
# ══════════════════════════════════════════════
X = df.drop(columns=['moteur', 'cycle', 'prix'])
y = df['prix']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train:{len(X_train)} Test:{len(X_test)}")

# ══════════════════════════════════════════════
# 5. ENTRAÎNER
# ══════════════════════════════════════════════
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"\n📊 Résultats :")
print(f"  RMSE : {rmse:.0f}€")
print(f"  R2   : {r2:.4f}")

# ══════════════════════════════════════════════
# 6. GRAPHIQUE — Prix réel vs prédit
# ══════════════════════════════════════════════
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color="steelblue")
plt.plot([500, 10000], [500, 10000], 'r--', label="Prediction parfaite")
plt.xlabel("Prix reel (€)")
plt.ylabel("Prix predit (€)")
plt.title("Prix reel vs Prix predit")
plt.legend()
plt.tight_layout()
plt.savefig("models/prix_revente.png")
print("✅ Graphique sauvegarde")

# ══════════════════════════════════════════════
# 7. SAUVEGARDER
# ══════════════════════════════════════════════
joblib.dump(model, 'models/model_prix.pkl')
joblib.dump(scaler, 'models/scaler_prix.pkl')
print("✅ Modele sauvegarde")