import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import plotly.graph_objects as go

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
Durante ao desenvolvimento, foram avaliadas duas abordagens:
- ARIMA
- Random Forest 
            
Seguimos com a escolha do Random Forest visto o ótimo retorno de resultado, como é possível visualizar na comparação abaixo:           

ARIMA:
- MAE = 20.21 → erro médio de ~20 dólares
- RMSE = 24.59 → quando erra, pode passar de 25 dólares de diferença.
- R² = -0.85 → pior que simplesmente usar a média histórica.            

Random Forest:            
- MAE = 1.33 → erro médio de apenas ~1 dólar.
- RMSE = 1.89 → mesmo os piores erros estão bem baixos.
- R² = 0.989 → explica 98,9% da variação da série.              

**Motivo da escolha do Random Forest Regressor:**
- Captura relações não lineares sem precisar de pré-processamento complexo.
- Robusto a outliers e ruídos nos dados históricos.
- Permite gerar previsões consistentes mesmo com dados temporais limitados.
            
**Agora é possível prever o preço do pretróleo pelos próximos 30 dias!**            
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

# Mantemos a coluna de preço como float para estilos
preco_coluna = 'Preço Previsto'
tabela_formatada[preco_coluna] = tabela_formatada[preco_coluna].astype(float)

# Estilizar tabela
st.dataframe(
    tabela_formatada.style
    .hide(axis="index")  # Oculta índice
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('color', '#333'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
    .background_gradient(subset=[preco_coluna], cmap='Oranges')  # Gradiente só funciona com float
    .format({preco_coluna: "{:.2f} USD"})  # Formatação visual como string
    , use_container_width=True
)

# ===========================
# Plotar gráfico interativo com Plotly
# ===========================
historico_visual = 60
y_recent = y.iloc[-historico_visual:]

fig = go.Figure()

# Histórico recente
fig.add_trace(go.Scatter(
    x=y_recent.index,
    y=y_recent,
    mode='lines',
    name=f'Preço Real (Últimos {historico_visual} dias)',
    line=dict(color='#1f77b4', width=2.5),
    hovertemplate='Data: %{x|%d/%m/%Y}<br>Preço Real: %{y:.2f} USD<extra></extra>'
))

# Previsão futura
fig.add_trace(go.Scatter(
    x=future_df['Data'],
    y=future_df['Preço Previsto'],
    mode='lines+markers',
    name=f'Previsão Futura ({n_dias} dias)',
    line=dict(color='#ff7f0e', width=2, dash='dash'),
    marker=dict(size=6),
    hovertemplate='Data: %{x|%d/%m/%Y}<br>Preço Previsto: %{y:.2f} USD<extra></extra>'
))

fig.update_layout(
    title='Preço do Petróleo: Histórico Recente x Previsão Futura',
    xaxis_title='Data',
    yaxis_title='Preço do Petróleo (USD)',
    template='plotly_white',
    hovermode='x unified'  # mantém hover sincronizado por data
)
st.plotly_chart(fig, use_container_width=True)

# ===========================
# Explicação sobre downloads
# ===========================
st.markdown("""
Você pode baixar os resultados gerados neste aplicativo:

- **CSV da Previsão:** contém as datas e os preços previstos para os próximos dias selecionados.
- **Gráfico Estático:** ilustra o histórico recente e a previsão futura, pronto para ser usado em apresentações ou relatórios.
""")

# ===========================
# Botão para download da tabela em CSV
# ===========================
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Baixar Previsão em Formato CSV",
    data=csv,
    file_name='previsao_petroleo.csv',
    mime='text/csv'
)

# ===========================
# Botão para download do gráfico
# ===========================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

# ===========================
# Criar gráfico estático para download
# ===========================
historico_visual = 60
y_recent = y.iloc[-historico_visual:]

fig, ax = plt.subplots(figsize=(16,7))

# Histórico recente
ax.plot(y_recent.index, y_recent, 
        label=f'Preço Real (Últimos {historico_visual} dias)',
        color='#1f77b4', linewidth=2.5)

# Previsão futura
ax.plot(future_df['Data'], future_df['Preço Previsto'], 
        label=f'Previsão Futura ({n_dias} dias)',
        color='#ff7f0e', linestyle='--', linewidth=2, marker='o', markersize=6)

# Layout
ax.set_facecolor('#ffffff')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('Data', fontsize=12)
ax.set_ylabel('Preço do Petróleo (USD)', fontsize=12)
ax.set_title('Preço do Petróleo: Histórico Recente x Previsão Futura', fontsize=16, pad=15)

# Formatação das datas
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
fig.autofmt_xdate(rotation=45)

# Legenda
ax.legend(fontsize=12, loc='upper left')

# Salvar imagem em BytesIO
buf = BytesIO()
fig.savefig(buf, format="png", bbox_inches='tight')
buf.seek(0)

# Botão de download
st.download_button(
    label="📥 Baixar Gráfico em Formato PNG",
    data=buf,
    file_name="grafico_previsao.png",
    mime="image/png"
)

plt.close(fig) 