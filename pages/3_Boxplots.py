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

# Substitua os valores "Free" por "0.00" na coluna 'Price'
df['Price'] = df['Price'].str.replace('Free', '0.00')

# Em seguida, converta a coluna 'Price' em tipo numérico
df['Price'] = df['Price'].str.replace('$', '').astype(float)

# Limpeza da coluna 'active_years' para obter os anos de início e término das séries
df['start_year'] = df['active_years'].str.extract(r'(\d{4})').astype(float)
# Preenche com NaN se houver valores inválidos
df['start_year'].fillna(np.nan, inplace=True)

# Tratando as inconsistências na coluna de Rating
# Normalizando todas as classificações para letras minúsculas
df['Rating'] = df['Rating'].str.lower()
# Agrupando os valores "no rating" e nulos como "não classificados"
df['Rating'].replace(['no rating', None], 'não classificados', inplace=True)

# Filtrando as 8 primeiras classificações de idade (ratings)
top_ratings = df['Rating'].value_counts().index[:8]
df_filtered = df[df['Rating'].isin(top_ratings)]

num_columns = df.select_dtypes(exclude='object').columns
cat_columns = df.select_dtypes(include='object').columns

st.subheader('Box Plots:')

# Cria intervalos de 10 anos
df['start_year_interval'] = ((df['start_year'] // 10) * 10).astype(int)

# Calcula a contagem de edições por intervalo
counts_by_interval = df['start_year_interval'].value_counts().sort_index()

# Visualização: Número de Edições por Ano de Início em intervalos de 10 anos
st.subheader('Número de Edições por Ano de Início (Intervalos de 10 Anos)')
plt.figure(figsize=(12, 6))
sns.barplot(x=counts_by_interval.index, y=counts_by_interval.values, color='blue')
plt.title('Número de Edições por Ano de Início (Intervalos de 10 Anos)')
plt.xlabel('Intervalo de Anos de Início')
plt.ylabel('Contagem')
plt.xticks(rotation=45)

# Corrige a ordem dos rótulos na legenda
start_years = counts_by_interval.index
end_years = start_years + 9
interval_labels = [f"{start}-{end}" for start, end in zip(start_years, end_years)]
plt.xticks(range(len(start_years)), interval_labels)
st.pyplot(plt)

# Visualização: Distribuição das Classificações de Idade (Rating)
st.subheader('Distribuição das 8 primeiras Classificações de Idade (Rating)')
plt.figure(figsize=(10, 6))
sns.countplot(x='Rating', data=df, order=df['Rating'].value_counts().index[:8], color='blue')
plt.title('Distribuição das 8 primeiras Classificações de Idade (Rating)')
plt.xlabel('Classificação de Idade')
plt.ylabel('Contagem')
st.pyplot(plt)

# Criando um gráfico de dispersão com Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Rating'], df_filtered['Price'], alpha=0.5, color='blue')
plt.title('Relação entre Preço e Classificação de Idade')
plt.xlabel('Classificação de Idade (Rating)')
plt.ylabel('Preço')
st.pyplot(plt)