import streamlit as st
import pandas as pd
import plotly.express as px

import functions


dataset = 'data/Marvel_Comics.parquet'
df = pd.read_parquet(dataset)


st.subheader('Dataframe:')
n, m = df.shape
st.write(f'<p style="font-size:130%">O dataset contém {n} linhas e {m} colunas.</p>', unsafe_allow_html=True)
st.dataframe(df)

st.subheader('Informações:')
c1, c2, c3 = st.columns([1, 2, 1])
c2.dataframe(functions.df_info(df))

st.subheader('Informações sobre Valores Ausentes:')
if df.isnull().sum().sum() == 0:
    st.write('Não há nenhum valor ausente no seu dataset.')
else:
    c1, c2, c3 = st.columns([0.5, 2, 0.5])
    c2.dataframe(functions.df_isnull(df), width=1500)
    functions.space(2)

st.subheader('Análise Descritiva:')
st.dataframe(df.describe())

num_columns = df.select_dtypes(exclude='object').columns
cat_columns = df.select_dtypes(include='object').columns

st.subheader('Distribuição de Colunas Numéricas:')
if len(num_columns) == 0:
        st.write('Não há colunas numéricas nos dados.')
else:
    for num_col in num_columns:
            fig = px.histogram(df, x=num_col)
            st.plotly_chart(fig, use_container_width=True)


df['Price'] = df['Price'].str.replace('Free', '0.00')
df['Price'] = df['Price'].str.replace('$', '').astype(float)
df = df[df['Price'] <= 50.0]

st.subheader('Gráficos de Contagem de Colunas Categóricas:')
if len(cat_columns) == 0:
        st.write('Não há colunas categóricas nos dados.')
else:
    for cat_col in cat_columns:
        fig = px.histogram(df, x=cat_col, color_discrete_sequence=['indianred'])
        st.plotly_chart(fig, use_container_width=True)

st.subheader('Análise de Outliers')
c1, c2, c3 = st.columns([1, 2, 1])
c2.dataframe(functions.number_of_outliers(df))
