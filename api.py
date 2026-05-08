from fastapi import FastAPI            # créer l'API
from pydantic import BaseModel         # valider les données
from typing import List                # type liste
import numpy as np                     # calculs mathématiques
import joblib                          # charger les modèles

# 1. CHARGER LES MODÈLES
model_rul   = joblib.load('models/model.pkl')           # prédit le RUL
scaler_rul  = joblib.load('models/scaler.pkl')          # normalise pour RUL
model_fv    = joblib.load('models/model_fin_vie.pkl')   # recycler/réparer/vendre
scaler_fv   = joblib.load('models/scaler_fin_vie.pkl')  # normalise pour fin de vie
model_prix  = joblib.load('models/model_prix.pkl')      # prédit le prix
scaler_prix = joblib.load('models/scaler_prix.pkl')     # normalise pour prix

# 2. CRÉER L'APPLICATION
app = FastAPI(title="Airbus Maintenance API")

# 3. STRUCTURE DES DONNÉES
# Dictionnaire vide que l'utilisateur remplit
class Donnees(BaseModel):
    features: List[float]   # liste de nombres décimaux

# 4. ROUTES

# Route d'accueil
@app.get("/")
def accueil():
    return {"message": "Airbus Maintenance API", "status": "ok"}

# Route de santé
@app.get("/health")
def sante():
    return {"status": "ok", "modeles": "charges"}

# Route principale — prédit RUL + décision + prix
@app.post("/predict")
def predire(data: Donnees):
    X = np.array(data.features).reshape(1, -1)  # convertit en tableau numpy

    # Prédit le RUL
    rul = model_rul.predict(scaler_rul.transform(X))[0]

    # Prédit la décision
    decision = model_fv.predict(scaler_fv.transform(X))[0]
    labels   = {0: "Recycler", 1: "Reparer", 2: "Vendre"}

    # Prédit le prix
    X_prix = np.append(X, rul).reshape(1, -1)   # ajoute RUL aux features
    prix   = model_prix.predict(scaler_prix.transform(X_prix))[0]

    return {
        "RUL"      : round(float(rul), 1),       # cycles restants
        "decision" : labels[int(decision)],       # recycler/réparer/vendre
        "prix"     : round(float(prix), 2),       # prix en €
    }