import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo do gráfico usando Seaborn
sns.set_style('darkgrid')

dataset = 'data/Marvel_Comics.parquet'
df = pd.read_parquet(dataset)

df['Price'] = df['Price'].str.replace('Free', '0.00')
df['Price'] = df['Price'].str.replace('$', '').astype(float)

# Limpeza da coluna 'active_years' para obter os anos de início e término das séries
df['start_year'] = df['active_years'].str.extract(r'(\d{4})').astype(float)
df['start_year'].fillna(np.nan, inplace=True)

# Tratando as inconsistências na coluna de Rating
df['Rating'] = df['Rating'].str.lower()
df['Rating'].replace(['no rating', None], 'não classificados', inplace=True)

# Filtrando as 8 primeiras classificações de idade (ratings)
top_ratings = df['Rating'].value_counts().index[:8]
df_filtered = df[df['Rating'].isin(top_ratings)]

num_columns = df.select_dtypes(exclude='object').columns
cat_columns = df.select_dtypes(include='object').columns

st.subheader('Box Plots:')
df['start_year_interval'] = ((df['start_year'] // 10) * 10).astype(int)
counts_by_interval = df['start_year_interval'].value_counts().sort_index()
st.subheader('Número de Edições por Ano de Início (Intervalos de 10 Anos)')

plt.figure(figsize=(12, 6))
sns.barplot(x=counts_by_interval.index, y=counts_by_interval.values, color='blue')
plt.title('Número de Edições por Ano de Início (Intervalos de 10 Anos)')
plt.xlabel('Intervalo de Anos de Início')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
st.pyplot(plt)

st.write('\n')

# Visualização: Distribuição das Classificações de Idade (Rating)
st.subheader('Distribuição das 8 primeiras Classificações de Idade (Rating)')
plt.figure(figsize=(10, 6))
sns.countplot(x='Rating', data=df, order=df['Rating'].value_counts().index[:8], color='blue')
plt.title('Distribuição das 8 primeiras Classificações de Idade (Rating)')
plt.xlabel('Classificação de Idade')
plt.ylabel('Contagem')
st.pyplot(plt)

st.write('\n')

# Criando um gráfico de dispersão
st.subheader('Gráfico de Dispersão: Preço vs. Classificação de Idade')
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Rating'], df_filtered['Price'], alpha=0.5, color='blue')
plt.title('Relação entre Preço e Classificação de Idade')
plt.xlabel('Classificação de Idade (Rating)')
plt.ylabel('Preço')
st.pyplot(plt)
