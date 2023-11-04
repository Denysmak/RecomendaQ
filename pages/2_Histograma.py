import streamlit as st
import pandas as pd
import plotly.express as px


dataset = 'data/Marvel_Comics.parquet'
df = pd.read_parquet(dataset)

st.subheader("Selecione a coluna que deseja visualizar:")
target_column = st.selectbox("", df.columns, index=len(df.columns) - 1)

st.subheader("Histograma da coluna")
fig = px.histogram(df, x=target_column)
c1, c2, c3 = st.columns([0.5, 2, 0.5])
c2.plotly_chart(fig)
