import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# ===========================
# Carregar dados
# ===========================
df = pd.read_excel("preco_petroleo.xlsx").sort_values("data")
df["data"] = pd.to_datetime(df["data"])
df = df.set_index("data")
y = df["preco_petroleo"].astype(float)

# ===========================
# Criar features
# ===========================
X = pd.DataFrame(index=df.index)
X['month'] = df.index.month
X['day'] = df.index.day
X['weekday'] = df.index.weekday

# Lags (1 a 7 dias anteriores)
for lag in range(1, 8):
    X[f'lag_{lag}'] = y.shift(lag)

# Médias móveis
X['rolling_mean_3'] = y.shift(1).rolling(window=3).mean()
X['rolling_mean_7'] = y.shift(1).rolling(window=7).mean()

# Remover linhas com NaN
X = X.dropna()
y_train = y.loc[X.index]

# ===========================
# Treinar Random Forest
# ===========================
model = RandomForestRegressor(
    n_estimators=200,   # menos árvores (arquivo menor)
    max_depth=10,       # limitar profundidade
    random_state=42,
    n_jobs=-1
)
model.fit(X, y_train)

# ===========================
# Salvar modelo comprimido
# ===========================
joblib.dump(model, "modelo.pkl", compress=3)
print("Modelo treinado e salvo como modelo.pkl (com compressão)")