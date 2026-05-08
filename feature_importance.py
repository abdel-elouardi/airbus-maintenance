import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. CHARGER LE MODÈLE
model = joblib.load('models/model.pkl')

# 2. RÉCUPÉRER LES COLONNES
cols = ['moteur', 'cycle', 'op1', 'op2', 'op3'] + [f'capteur{i}' for i in range(1,22)]
df = pd.read_csv('data/raw/train_FD001.txt', sep='\s+', header=None, names=cols)
df.columns = df.columns.str.strip()
cols_constantes = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=cols_constantes)
cols_capteurs = [col for col in df.columns if 'capteur' in col]
cols_a_supprimer = []
corr = df[cols_capteurs].corr().abs()
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if corr.loc[corr.columns[i], corr.columns[j]] > 0.9:
            cols_a_supprimer.append(corr.columns[j])
df = df.drop(columns=list(set(cols_a_supprimer)))
cols_features = [col for col in df.columns if col not in ['moteur', 'cycle']]

# 3. IMPORTANCE
importances = pd.Series(model.feature_importances_, index=cols_features)
importances = importances.sort_values(ascending=False)
print("📊 Pieces a surveiller :")
for nom, val in importances.items():
    print(f"  {nom:15s} : {val*100:.1f}%")

# 4. GRAPHIQUE
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values * 100, y=importances.index, hue=importances.index, palette="Blues_r", legend=False)
plt.title("Pieces a surveiller en priorite")
plt.xlabel("Importance (%)")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
print("✅ Graphique sauvegardé")