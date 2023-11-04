import streamlit as st
import pandas as pd
import plotly.express as px

dataset = 'Marvel_Comics.parquet'
df = pd.read_parquet(dataset)

num_columns = df.select_dtypes(exclude='object').columns
cat_columns = df.select_dtypes(include='object').columns
target_column = st.selectbox("", df.columns, index=len(df.columns) - 1)


st.subheader('Variância do Alvo com Colunas Categóricas:')
df_1 = df.dropna()
high_cardi_columns = []
normal_cardi_columns = []

for cat_col in cat_columns:
    if df[cat_col].nunique() > df.shape[0] / 10:
        high_cardi_columns.append(cat_col)
    else:
        normal_cardi_columns.append(cat_col)

if len(normal_cardi_columns) == 0:
        st.write('Não há colunas categóricas com cardinalidade normal nos dados.')
else:
    model_type = 'Regressão'  # Pode escolher 'Regressão' ou 'Classificação'
    for cat_col in normal_cardi_columns:
        if model_type == 'Regressão':
            fig = px.box(df_1, y=target_column, color=cat_col)
        else:
            fig = px.histogram(df_1, color=cat_col, x=target_column)
        st.plotly_chart(fig, use_container_width=True)

    if high_cardi_columns:
        if len(high_cardi_columns) == 1:
                st.subheader('A seguinte coluna tem alta cardinalidade, por isso o gráfico de caixa não foi plotado:')
        else:
                st.subheader('As seguintes colunas têm alta cardinalidade, por isso os gráficos de caixa não foram plotados:')
        for high_cardi_col in high_cardi_columns:
                st.write(high_cardi_col)

        st.write('<p style="font-size:140%">Deseja plotar mesmo assim?</p>', unsafe_allow_html=True)
        answer = st.selectbox("", ('Não', 'Sim'))

        if answer == 'Sim':
            for high_cardi_col in high_cardi_columns:
                fig = px.box(df_1, y=target_column, color=high_cardi_col)
                st.plotly_chart(fig, use_container_width=True)
                