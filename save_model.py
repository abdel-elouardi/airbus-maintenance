import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Charger et nettoyer
cols = ['moteur', 'cycle', 'op1', 'op2', 'op3'] + [f'capteur{i}' for i in range(1,22)]
df = pd.read_csv('data/raw/train_FD001.txt', sep='\s+', header=None, names=cols)
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.drop(columns=df.columns[df.isnull().mean() > 0.75])
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median() if abs(df[col].skew()) > 1 else df[col].mean())
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

# Normaliser
cols_a_normaliser = [col for col in df.columns if col not in ['moteur', 'cycle']]
scaler = StandardScaler()
df[cols_a_normaliser] = scaler.fit_transform(df[cols_a_normaliser])

# RUL
max_cycles = df.groupby('moteur')['cycle'].max()
df['cycle_max'] = df['moteur'].map(max_cycles)
df['RUL'] = (df['cycle_max'] - df['cycle']).clip(upper=125)
df = df.drop(columns=['cycle_max'])

# Split
X = df.drop(columns=['moteur', 'cycle', 'RUL'])
y = df['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner et sauvegarder
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Modèle sauvegardé → models/model.pkl")
print("✅ Scaler sauvegardé → models/scaler.pkl")
