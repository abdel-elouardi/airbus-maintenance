import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ══════════════════════════════════════════════
# 1. CHARGER ET NETTOYER
# Même nettoyage que main.py
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
print(f"✅ Données chargées et nettoyées : {df.shape}")

# ══════════════════════════════════════════════
# 2. CALCULER LE RUL
# RUL = combien de cycles restants avant la panne
# On plafonne à 125 pour améliorer le modèle
# ══════════════════════════════════════════════
max_cycles = df.groupby('moteur')['cycle'].max()
df['cycle_max'] = df['moteur'].map(max_cycles)
df['RUL'] = (df['cycle_max'] - df['cycle']).clip(upper=125)
df = df.drop(columns=['cycle_max'])
print(f"✅ RUL calculé — Min:{df['RUL'].min()} Max:{df['RUL'].max()}")

# ══════════════════════════════════════════════
# 3. NORMALISER
# Met toutes les colonnes à la même échelle
# ══════════════════════════════════════════════
cols_features = [col for col in df.columns if col not in ['moteur', 'cycle', 'RUL']]
scaler = StandardScaler()
df[cols_features] = scaler.fit_transform(df[cols_features])
print("✅ Normalisation terminée")

# ══════════════════════════════════════════════
# 4. CRÉER LES SÉQUENCES
# LSTM a besoin de séquences de N cycles consécutifs
# On prend les 30 derniers cycles pour prédire le RUL
# Ex: cycles 1→30 → RUL=95, cycles 2→31 → RUL=94...
# ══════════════════════════════════════════════
SEQUENCE = 30
X_seq, y_seq = [], []
for moteur in df['moteur'].unique():
    df_m = df[df['moteur'] == moteur].reset_index(drop=True)
    for i in range(SEQUENCE, len(df_m)):
        X_seq.append(df_m[cols_features].iloc[i-SEQUENCE:i].values)
        y_seq.append(df_m['RUL'].iloc[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(f"✅ Séquences créées : {X_seq.shape}")
# X_seq.shape = (nb_séquences, 30 cycles, nb_features)

# ══════════════════════════════════════════════
# 5. SPLIT TRAIN / TEST
# 80% pour entraîner, 20% pour tester
# Convertit en tenseurs PyTorch
# ══════════════════════════════════════════════
split = int(0.8 * len(X_seq))
X_train = torch.FloatTensor(X_seq[:split])
X_test  = torch.FloatTensor(X_seq[split:])
y_train = torch.FloatTensor(y_seq[:split])
y_test  = torch.FloatTensor(y_seq[split:])
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
print(f"✅ Train:{len(X_train)} Test:{len(X_test)}")

# ══════════════════════════════════════════════
# 6. MODÈLE LSTM
# input_dim  = nombre de features (capteurs)
# hidden_dim = taille de la mémoire cachée
# num_layers = nombre de couches LSTM empilées
# fc         = couche finale qui prédit le RUL
# ══════════════════════════════════════════════
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        # LSTM : lit la séquence et garde une mémoire
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2)
        # Couche finale : transforme la mémoire en RUL
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # On prend seulement le dernier pas de temps
        return self.fc(out[:, -1, :]).squeeze()

# ══════════════════════════════════════════════
# 7. ENTRAÎNEMENT
# 50 epochs — le modèle apprend à chaque passage
# Adam optimizer — ajuste les poids du réseau
# MSELoss — mesure l'erreur de prédiction
# ══════════════════════════════════════════════
device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model     = LSTM(input_dim=X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
print(f"🚀 Entraînement LSTM sur {device}...")

for epoch in range(1, 51):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()        # Remet les gradients à zéro
        loss = criterion(model(X_b), y_b)  # Calcule l'erreur
        loss.backward()              # Rétropropagation
        optimizer.step()             # Met à jour les poids
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}/50 | Loss: {loss.item():.4f}")

# ══════════════════════════════════════════════
# 8. ÉVALUATION
# Compare LSTM vs Gradient Boosting
# ══════════════════════════════════════════════
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_test.numpy(), y_pred))
r2   = r2_score(y_test.numpy(), y_pred)
print(f"\n📊 Résultats :")
print(f"  LSTM              — RMSE:{rmse:.2f} | R2:{r2:.4f}")
print(f"  Gradient Boosting — RMSE:18.90 | R2:0.79")