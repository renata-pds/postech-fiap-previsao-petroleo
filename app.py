import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Previsão do Preço do Petróleo", layout="wide")
st.title("⛽ Previsão do Preço do Petróleo (USD)")
st.caption("Aplicação Streamlit usando Random Forest — carrega modelo treinado ou permite re-treinamento rápido.")

# --------- Helpers ---------
def load_default_data():
    local = Path('preco_petroleo.xlsx')
    if local.exists():
        return pd.read_excel(local)
    return None

def prep_features(y):
    X = pd.DataFrame()
    for lag in range(1, 8):
        X[f"lag_{lag}"] = y.shift(lag)
    X["rolling_mean_3"] = y.shift(1).rolling(3).mean()
    X["rolling_mean_7"] = y.shift(1).rolling(7).mean()
    X = X.dropna()
    y_aligned = y[X.index]
    return X, y_aligned

def train_rf(y):
    X, y_train = prep_features(y)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y_train)
    return rf

def forecast_future(model, y, n_days=30):
    last_price = y.iloc[-7:].tolist()
    future_preds = []
    future_dates = []

    for i in range(n_days):
        next_date = y.index[-1] + pd.Timedelta(days=i+1)
        future_dates.append(next_date)

        new_row = {}
        for lag in range(1, 8):
            new_row[f"lag_{lag}"] = last_price[-lag]
        new_row['rolling_mean_3'] = np.mean(last_price[-3:])
        new_row['rolling_mean_7'] = np.mean(last_price[-7:])

        new_df = pd.DataFrame(new_row, index=[next_date])
        pred = model.predict(new_df)[0]
        future_preds.append(pred)
        last_price.append(pred)

    return pd.DataFrame({'data': future_dates, 'preco_previsto': future_preds})

def save_model(model, path: Path):
    joblib.dump(model, path)

def load_model(path: Path):
    return joblib.load(path)

# --------- Sidebar ---------
st.sidebar.header("Configurações")
n_days = st.sidebar.number_input("Horizonte de previsão (dias):", min_value=7, max_value=90, value=30, step=1)
retrain = st.sidebar.checkbox("Treinar/Re-treinar modelo com os dados carregados", value=False)

uploaded = st.file_uploader("Faça upload de um arquivo Excel com colunas 'data' e 'preco_petroleo' (ou deixe em branco para usar o padrão)", type=["xlsx"])

# --------- Data loading ---------
if uploaded is not None:
    df = pd.read_excel(uploaded)
elif load_default_data() is not None:
    df = load_default_data()
else:
    st.info("Nenhum dado fornecido e arquivo padrão não encontrado. Por favor, forneça os dados.")
    st.stop()

st.subheader("Amostra dos Dados")
st.dataframe(df.head(20))

df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data').set_index('data')
y = df['preco_petroleo'].astype(float)

# --------- Model loading / training ---------
model_path = Path("modelo.pkl")
model = None

if retrain:
    with st.spinner("Treinando modelo Random Forest..."):
        model = train_rf(y)
        save_model(model, model_path)
    st.success("✅ Modelo treinado com sucesso!")
else:
    if model_path.exists():
        model = load_model(model_path)
        st.success(f"✅ Modelo carregado de '{model_path.name}'.")
    else:
        st.warning("Modelo não encontrado. Treinando modelo rapidamente agora...")
        model = train_rf(y)
        save_model(model, model_path)
        st.success("✅ Modelo treinado com sucesso!")

# --------- Forecast ---------
future_df = forecast_future(model, y, n_days=int(n_days))

st.subheader("Previsão de Preços Futuros")
st.write(f"Horizonte: **{n_days} dias** | Última data de treino: **{y.index.max().date()}**")

# Plot histórico + previsão
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y.index, y, label="Histórico")
ax.plot(future_df['data'], future_df['preco_previsto'], label="Previsão", linestyle="--", color='red')
ax.set_xlabel("Data")
ax.set_ylabel("Preço do Petróleo (USD)")
ax.set_title("Histórico x Previsão")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Mostrar tabela
st.subheader("Tabela de Previsão")
st.dataframe(future_df.set_index('data'))

# Download CSV
csv = future_df.to_csv(index=False).encode("utf-8")
st.download_button("Baixar previsão em CSV", data=csv, file_name="previsao_petroleo.csv", mime="text/csv")

st.divider()
with st.expander("ℹ️ Como este modelo funciona"):
    st.markdown(
        "- Pré-processamento: criação de lags e médias móveis.\n"
        "- Modelo: Random Forest Regressor treinado com os dados fornecidos.\n"
        "- O app permite re-treinar rapidamente ou usar modelo salvo para previsão."
    )