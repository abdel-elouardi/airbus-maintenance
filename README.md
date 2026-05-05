# ✈️ Airbus — Maintenance Prédictive

Prédiction de la durée de vie restante (RUL) des moteurs d'avion.

## 📊 Dataset
- NASA CMAPSS — données réelles de capteurs moteurs
- 20631 mesures
- 21 capteurs : température, pression, vibrations

## 🤖 Résultats

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
| LSTM | 17.94 🏆 | 0.82 |

## 🏆 Conclusion
LSTM bat Gradient Boosting — RMSE 17.94 vs 18.90

## 🚀 Lancement
pip install -r requirements.txt
python main.py   # Sklearn
python lstm.py   # PyTorch LSTM

## 🛠️ Technologies
Python 3.11 | Scikit-learn | XGBoost | LightGBM | PyTorch LSTM