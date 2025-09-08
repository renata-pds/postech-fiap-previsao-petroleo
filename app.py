import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# ==============================
# Carregar modelo treinado
# ==============================
model = joblib.load("modelo.pkl")

# ==============================
# Função para prever preços futuros
# ==============================
def prever_precos_futuros(model, y, n_days=30):
    last_price = y.iloc[-7:].tolist()
    future_preds = []
    future_dates = []

    for i in range(n_days):
        next_date = y.index[-1] + pd.Timedelta(days=i+1)
        future_dates.append(next_date)

        # Features do próximo dia
        new_row = {
            'month': next_date.month,
            'day': next_date.day,
            'weekday': next_date.weekday()
        }
        # Lags
        for lag in range(1, 8):
            new_row[f'lag_{lag}'] = last_price[-lag]
        # Médias móveis
        new_row['rolling_mean_3'] = np.mean(last_price[-3:])
        new_row['rolling_mean_7'] = np.mean(last_price[-7:])

        new_df = pd.DataFrame(new_row, index=[next_date])

        # Previsão
        pred = model.predict(new_df)[0]
        future_preds.append(pred)
        last_price.append(pred)

    return pd.DataFrame({'data': future_dates, 'preco_predito': future_preds})

# ==============================
# Interface Streamlit
# ==============================
st.set_page_config(page_title="Previsão do Preço do Petróleo", layout="wide")
st.title("⛽ Previsão do Preço do Petróleo em USD")

# Upload do arquivo de histórico
uploaded_file = st.file_uploader("Faça upload do arquivo de histórico (Excel)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file).sort_values("data")
    df["data"] = pd.to_datetime(df["data"])
    df = df.set_index("data")
    y = df["preco_petroleo"].astype(float)

    # Seleção de horizonte de previsão
    n_days = st.slider("Número de dias futuros para prever:", 7, 90, 30)

    # Previsão
    future_df = prever_precos_futuros(model, y, n_days=n_days)

    # Plot histórico + previsão
    st.subheader("Histórico e Previsão")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y.index, y, label="Preço Real")
    ax.plot(future_df["data"], future_df["preco_predito"], label="Previsão Futura", linestyle="--", color="red")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço do Petróleo (USD)")
    ax.set_title("Preço do Petróleo: Histórico x Previsão")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Mostrar tabela de previsões
    st.subheader("Tabela de Previsões Futuras")
    st.dataframe(future_df.set_index("data"))
else:
    st.info("Por favor, faça upload de um arquivo Excel com as colunas: `data` e `preco_petroleo`.")