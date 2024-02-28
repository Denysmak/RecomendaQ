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
df['Rating'] = df['Rating'].str.lower().str.replace('rated', '').str.strip()
df['Rating'].replace(['no rating', None], 'não classificados', inplace=True)

rating_mapping = {
    't+': 13,
    't': 10,
    'parental advisory': 15,
    'all ages': 0,
    'marvel psr': 15,
    'a': 9,
    'explicit content': 18
}

# Filtrando as 8 primeiras classificações de idade (ratings)
top_ratings = df['Rating'].value_counts().index[:8]
df_filtered = df[df['Rating'].isin(top_ratings)]
df_filtered_with_unrated = df[df['Rating'].isin(top_ratings)]

# Excluindo as não classificadas na versão sem não classificadas
df_filtered_without_unrated = df[df['Rating'].isin(top_ratings) & (df['Rating'] != 'não classificados')]
# Remover linhas com valores NaN na coluna 'Price'
df_filtered = df_filtered.dropna(subset=['Price'])
# Filtrar o DataFrame para incluir apenas anos a partir de 1930
df_filtered = df_filtered[df_filtered['start_year'] >= 1930]


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
st.subheader('Distribuição das principais Classificações indicativas (Rating)')
plt.figure(figsize=(10, 6))
sns.countplot(x='Rating', data=df_filtered_with_unrated, order=top_ratings, color='blue')
plt.title('Distribuição das 8 primeiras Classificações de Idade (Rating)')
plt.xlabel('Classificação de Idade')
plt.ylabel('Contagem')
st.pyplot(plt)

st.write('\n')

# Versão 2: Excluindo as Não Classificadas
st.subheader('Distribuição das principais Classificações indicativas (Excluindo Não Classificadas)')
plt.figure(figsize=(10, 6))
sns.countplot(x='Rating', data=df_filtered_without_unrated, order=top_ratings, color='blue')
plt.title('Distribuição das 8 primeiras Classificações de Idade (Excluindo Não Classificadas)')
plt.xlabel('Classificação de Idade')
plt.ylabel('Contagem')
st.pyplot(plt)

st.write('\n')

# Gráfico de dispersão vendo o preço de acordo com os anos separando os cluster como rating
st.subheader('Gráfico de dispersão mostrando o preço peço ano de inicío de acordo com a classificação indicativa')
scatter_fig = px.scatter(df_filtered_without_unrated, x='start_year', y='Price', color='Rating',
                         title='Relação entre Preço e Ano de Lançamento por Classificação Indicativa',
                         labels={'start_year': 'Ano de Lançamento', 'Price': 'Preço', 'Rating': 'Classificação Indicativa'})
scatter_fig.update_layout(title_font_size=20, title_font_family="Arial", title_font_color="black",
                          xaxis_title_font_size=16, yaxis_title_font_size=16,
                          xaxis=dict(title='Ano de Lançamento'),
                          yaxis=dict(title='Preço'))
st.plotly_chart(scatter_fig)

# Criando um gráfico de dispersão
st.subheader('Gráfico de Dispersão: Preço vs Classificação de Idade')
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Rating'], df_filtered['Price'], alpha=0.5, color='blue')
plt.title('Relação entre Preço e Classificação de Idade')
plt.xlabel('Classificação de Idade (Rating)')
plt.ylabel('Preço')
st.pyplot(plt)

# Criando um gráfico de bolha
st.subheader('Gráfico de Bolha: Preço vs Classificação de Idade')
bubble_fig = px.scatter(df_filtered, x='Rating', y='Price', size='Price', color='Rating',
                        title='Gráfico de Bolha: Preço vs Classificação de Idade',
                        labels={'Rating': 'Classificação de Idade', 'Price': 'Preço'})
st.plotly_chart(bubble_fig)

# Criar um gráfico de caixa
st.subheader('Gráfico de Caixa: Distribuição de Preços por Ano de Início da Série')
boxplot_fig = px.box(df_filtered, x='start_year', y='Price',
                     title='Gráfico de Caixa: Distribuição de Preços por Ano de Início da Série',
                     labels={'start_year': 'Ano de Início da Série', 'Price': 'Preço'})
st.plotly_chart(boxplot_fig)

# Criar um gráfico de violino
st.subheader('Gráfico de Violino: Distribuição de Preços por Ano de Início da Série')
violin_fig = px.violin(df_filtered, x='start_year', y='Price', box=True, points="all",
                       title='Gráfico de Violino: Distribuição de Preços por Ano de Início da Série',
                       labels={'start_year': 'Ano de Início da Série', 'Price': 'Preço'})
st.plotly_chart(violin_fig)