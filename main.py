import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. CHARGER
cols = ['moteur', 'cycle', 'op1', 'op2', 'op3'] + [f'capteur{i}' for i in range(1,22)]
df = pd.read_csv('data/raw/train_FD001.txt', sep='\s+', header=None, names=cols)
print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# 2. NETTOYAGE

# Espaces
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Types
for col in df.columns:
    if df[col].dtype == object:
        try: df[col] = pd.to_numeric(df[col])
        except: pass

# Doublons
avant = len(df)
df = df.drop_duplicates()
print(f"✅ Doublons supprimés : {avant - len(df)}")

# Colonnes > 75% vides
cols_vides = df.columns[df.isnull().mean() > 0.75].tolist()
df = df.drop(columns=cols_vides)
if cols_vides:
    print(f"✅ Colonnes supprimées : {cols_vides}")

# Valeurs manquantes
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median() if abs(df[col].skew()) > 1 else df[col].mean())
print("✅ Valeurs manquantes remplacées")

# Colonnes constantes
cols_constantes = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=cols_constantes)
print(f"✅ Colonnes constantes supprimées : {cols_constantes}")

# Skewness
cols_capteurs = [col for col in df.columns if 'capteur' in col]
for col in cols_capteurs:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col] - df[col].min())
print("✅ Skewness corrigée")

# Corrélation > 0.9
corr = df[cols_capteurs].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
cols_corr = [c for c in upper.columns if any(upper[c] > 0.9)]
df = df.drop(columns=cols_corr)
print(f"✅ Colonnes très corrélées supprimées : {cols_corr}")

# Normalisation
cols_a_normaliser = [col for col in df.columns if col not in ['moteur', 'cycle']]
scaler = StandardScaler()
df[cols_a_normaliser] = scaler.fit_transform(df[cols_a_normaliser])
print("✅ Normalisation terminée")

print(f"\n📊 Après nettoyage : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Cycle max de chaque moteur = cycle de panne
max_cycles = df.groupby('moteur')['cycle'].max()

# Ajouter au dataset
df['cycle_max'] = df['moteur'].map(max_cycles)

# RUL = cycle_max - cycle_actuel
df['RUL'] = df['cycle_max'] - df['cycle']
# RUL plafonné à 125 — au-delà le moteur est "comme neuf"
# Technique standard pour le dataset NASA CMAPSS
df['RUL'] = df['RUL'].clip(upper=125)
print(f"✅ RUL plafonné — Min:{df['RUL'].min()} Max:{df['RUL'].max()} Moyenne:{df['RUL'].mean():.0f}")
# Supprimer cycle_max
df = df.drop(columns=['cycle_max'])

print(f"✅ RUL — Min:{df['RUL'].min()} Max:{df['RUL'].max()} Moyenne:{df['RUL'].mean():.0f}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 4. SPLIT
X = df.drop(columns=['moteur', 'cycle', 'RUL'])
y = df['RUL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"✅ Train:{len(X_train)} Test:{len(X_test)}")

# 5. MODÈLES
models = {
    "Linear Regression":   LinearRegression(),
    "Ridge":               Ridge(),
    "Random Forest":       RandomForestRegressor(random_state=42),
    "Gradient Boosting":   GradientBoostingRegressor(random_state=42),
    "KNN":                 KNeighborsRegressor(),
    "XGBoost":             XGBRegressor(random_state=42),
    "LightGBM":            LGBMRegressor(random_state=42, verbose=-1),
}

resultats = {}
for nom, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    resultats[nom] = {"rmse": rmse, "r2": r2}
    print(f"  {nom:25s} | RMSE:{rmse:.2f} | R2:{r2:.4f}")

# 6. MEILLEUR MODÈLE
meilleur = min(resultats, key=lambda x: resultats[x]['rmse'])
print(f"\n🏆 Meilleur : {meilleur} (RMSE:{resultats[meilleur]['rmse']:.2f})")

# 7. GRAPHIQUE
plt.figure(figsize=(10, 5))
noms = list(resultats.keys())
rmses = [resultats[n]['rmse'] for n in noms]
plt.barh(noms, rmses, color="steelblue")
plt.title("Comparaison des modèles — RMSE")
plt.tight_layout()
plt.savefig("models/comparaison.png")
print("✅ Graphique sauvegardé")