import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib


# ===========================
# Configuração da página
# ===========================
st.set_page_config(page_title="Previsão do Preço do Petróleo", layout="wide")

# ===========================
# Barra lateral fixa
# ===========================
st.sidebar.title("Desenvolvedora:")
st.sidebar.write("Renata Paes da Silva - RM359515")

# ===========================
# Título e introdução
# ===========================
st.title("⛽ Previsão do Preço do Petróleo (USD)")
st.markdown("""
Este aplicativo utiliza um **Random Forest Regressor** para prever o preço do petróleo.  

**Motivo da escolha do Random Forest:**
- Captura relações não lineares sem precisar de pré-processamento complexo.
- Robusto a outliers e ruídos nos dados históricos.
- Permite gerar previsões consistentes mesmo com dados temporais limitados.
""")

# ===========================
# Carregar modelo e dados
# ===========================
model = joblib.load("modelo.pkl")
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

for lag in range(1, 8):
    X[f'lag_{lag}'] = y.shift(lag)

X['rolling_mean_3'] = y.shift(1).rolling(window=3).mean()
X['rolling_mean_7'] = y.shift(1).rolling(window=7).mean()
X = X.dropna()

# ===========================
# Função de previsão
# ===========================
def prever_precos_futuros(model, X, y, n_days=30):
    last_price = y.iloc[-7:].tolist()
    future_preds = []
    future_dates = []

    for i in range(n_days):
        next_date = X.index[-1] + pd.Timedelta(days=i+1)
        future_dates.append(next_date)

        new_row = {
            'month': next_date.month,
            'day': next_date.day,
            'weekday': next_date.weekday()
        }
        for lag in range(1, 8):
            new_row[f'lag_{lag}'] = last_price[-lag]
        new_row['rolling_mean_3'] = np.mean(last_price[-3:])
        new_row['rolling_mean_7'] = np.mean(last_price[-7:])

        new_df = pd.DataFrame(new_row, index=[next_date])
        pred = model.predict(new_df)[0]
        future_preds.append(pred)
        last_price.append(pred)

    future_df = pd.DataFrame({
        'Data': pd.to_datetime(future_dates),
        'Preço Previsto': np.array(future_preds)
    })
    return future_df

# ===========================
# Slider para escolher quantidade de dias
# ===========================
n_dias = st.slider("Selecione o número de dias que deseja fazer a previsão", min_value=1, max_value=30, value=1)

# ===========================
# Gerar previsão
# ===========================
future_df = prever_precos_futuros(model, X, y, n_days=n_dias)

# ===========================
# Mostrar indicadores
# ===========================
st.subheader("Indicadores da Previsão")
col1, col2, col3 = st.columns(3)
col1.metric("Último Preço Real", f"{y.iloc[-1]:.2f}")
col2.metric(f"Preço Previsto para o dia {future_df['Data'].dt.strftime('%d/%m/%Y').iloc[-1]}", f"{future_df['Preço Previsto'].iloc[-1]:.2f}")
col3.metric("Média da Previsão", f"{future_df['Preço Previsto'].mean():.2f}")

# ===========================
# Mostrar tabela com previsão
# ===========================
st.subheader("Tabela de Previsão")
tabela_formatada = future_df.copy()
tabela_formatada['Data'] = tabela_formatada['Data'].dt.strftime('%d/%m/%Y')
tabela_formatada['Preço Previsto'] = tabela_formatada['Preço Previsto'].apply(lambda x: f"{x:.2f} USD")

# Exibir sem índice numérico
st.dataframe(tabela_formatada.style.hide(axis="index"), use_container_width=True)

# ===========================
# Plotar gráfico
# ===========================
historico_visual = 60
y_recent = y.iloc[-historico_visual:]

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(y_recent.index, y_recent, label=f'Preço Real (Últimos {historico_visual} dias)', color='blue', linewidth=2)
ax.plot(future_df['Data'], future_df['Preço Previsto'], label=f'Previsão Futura ({n_dias} dias)', color='red', linestyle='--', linewidth=2, marker='o')

# Formatando as datas
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

ax.set_xlabel('Data')
ax.set_ylabel('Preço do Petróleo')
ax.set_title('Preço do Petróleo: Histórico Recente x Previsão Futura')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

st.pyplot(fig)

