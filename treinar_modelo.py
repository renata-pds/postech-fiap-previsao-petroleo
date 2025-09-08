import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# ===========================
# Carregar dados
# ===========================
df = pd.read_excel("preco_petroleo.xlsx")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").set_index("data")
y = df["preco_petroleo"].astype(float)

# ===========================
# Criar features simples (lags e médias móveis)
# ===========================
X = pd.DataFrame()
for lag in range(1, 8):
    X[f"lag_{lag}"] = y.shift(lag)
X["rolling_mean_3"] = y.shift(1).rolling(3).mean()
X["rolling_mean_7"] = y.shift(1).rolling(7).mean()

# Remover linhas com NaN
X = X.dropna()
y_train = y[X.index]

# ===========================
# Treinar Random Forest
# ===========================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y_train)

# ===========================
# Salvar modelo
# ===========================
joblib.dump(rf, "modelo.pkl")
print("Modelo treinado e salvo como modelo.pkl")