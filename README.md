# ✈️ Airbus — Système de Maintenance Prédictive

Système ML complet de gestion du cycle de vie des moteurs d'avion.
Basé sur des données réelles de capteurs NASA CMAPSS.

## 📊 Dataset
- **Source** : NASA CMAPSS
- **Taille** : 20 631 mesures
- **Capteurs** : 21 capteurs (température, pression, vibrations)
- **Moteurs** : 100 moteurs

## 🤖 Modules

| Module | Fichier | Résultat |
|--------|---------|---------|
| Maintenance prédictive Sklearn | main.py | RMSE 18.90 |
| Maintenance prédictive LSTM | lstm.py | RMSE 17.40 🏆 |
| Pièces critiques | feature_importance.py | capteur11 = 35% |
| Classification fin de vie | classification_fin_vie.py | 85% accuracy |
| Prix de revente | prix_revente.py | R2 = 0.94 |
| Gestion du stock | gestion_stock.py | 89 pièces à vendre |
| API FastAPI | api.py | /predict |

## 🏆 Résultats

### Sklearn
| Modèle | RMSE | R2 |
|--------|------|----|
| Gradient Boosting | 18.90 🏆 | 0.79 |
| LightGBM | 18.97 | 0.79 |
| Random Forest | 19.07 | 0.79 |
| XGBoost | 19.91 | 0.77 |
| KNN | 20.52 | 0.75 |
| Linear Regression | 21.69 | 0.72 |

### PyTorch LSTM
| Modèle | RMSE | R2 |
|--------|------|----|
| LSTM | 17.40 🏆 | 0.83 |

## 🔧 Pièces critiques
| Capteur | Importance |
|---------|-----------|
| capteur11 | 35.4% 🚨 |
| capteur4  | 21.1% |
| capteur9  | 13.9% |
| capteur12 | 13.8% |

## 🚀 Lancement

```bash
# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données
python3 -c "import kagglehub; kagglehub.dataset_download('behrad3d/nasa-cmaps')"

# Entraîner les modèles
python3 main.py
python3 lstm.py
python3 feature_importance.py
python3 classification_fin_vie.py
python3 prix_revente.py
python3 gestion_stock.py

# Lancer l'API
uvicorn api:app --port 8002

# Tester l'API
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [120.0, 0.5, 641.0, 14.62, 21.61, 554.0, 2388.0, 9046.0, 1.3, 47.0, 521.0, 2388.0, 8138.0, 8.4, 0.03, 392.0]}'
```

## 🛠️ Technologies
Python 3.11 | Scikit-learn | PyTorch LSTM | XGBoost | LightGBM | FastAPI | Docker