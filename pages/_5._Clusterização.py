import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
from sklearn.decomposition import PCA

# Carregando o conjunto de dados
dataset = 'data/Marvel_Comics.parquet'
df = pd.read_parquet(dataset)

# Limpeza e pré-processamento dos dados
df['Price'] = df['Price'].str.replace('Free', '0.00').str.replace('$', '').astype(float)

def adjust_price_for_inflation(original_price, original_year, current_year=None, inflation_rate=0.034):
    if current_year is None:
        current_year = datetime.now().year
    years_passed = current_year - original_year
    inflation_multiplier = (1 + inflation_rate) ** years_passed
    adjusted_price = original_price * inflation_multiplier
    return adjusted_price

df['start_year'] = df['active_years'].str.extract(r'(\d{4})').astype(float)
df['start_year'] = df['start_year'].where(df['start_year'] >= 1800)  # Tratando anos inválidos
df['Price'] = df.apply(lambda row: adjust_price_for_inflation(row['Price'], row['start_year']), axis=1)
df['Rating'] = df['Rating'].str.lower().str.replace('rated', '').str.strip()
df['Rating'].replace(['no rating', None], 'não classificados', inplace=True)

top_ratings = df['Rating'].value_counts().index[:8]
df = df[df['Rating'].isin(top_ratings)]

rating_mapping = {
    't+': 13,
    't': 10,
    'parental advisory': 15,
    'all ages': 0,
    'marvel psr': 15,
    'a': 9,
    'explicit content': 18
}

format_mapping = {
        #' Infinite Comic': 0,
        ' Comic': 1,
        ' Digest': 2,
        ' None': 1,
        ' Digital Comic': 4,
        ' Trade Paperback': 5,
        ' Hardcover': 6,
        ' MAGAZINE': 7,
        ' Magazine': 7,
        ' comic': 1,
        ' DIGITAL COMIC': 4,
        ' Graphic Novel': 11,
    }

df['Rating'] = df['Rating'].map(rating_mapping)
df.loc[df['Rating'] == 'não classificados', 'Rating'] = np.nan

df['Format'] = df['Format'].map(format_mapping)
df['Format'].fillna(-1, inplace=True)

# Interface Streamlit
st.title('Agrupamento (Clustering) com a base do Marvel Comics')

# Definição das colunas para clustering
clustering_cols_opts = ['Price', 'Rating', 'start_year', 'Format']
clustering_cols = st.multiselect('Selecione as colunas para clustering', clustering_cols_opts, default=clustering_cols_opts[:2])

# Seleção do algoritmo de clustering
algorithm = st.selectbox('Selecione o algoritmo de clustering', ['KMeans', 'DBSCAN'])

# Exibindo média e desvio padrão das colunas por cluster
if algorithm == 'KMeans':
    n_clusters = st.slider('Número de Clusters (KMeans)', min_value=2, max_value=10, value=3)

    # Pré-processamento dos dados
    df_processed = df[clustering_cols].dropna()

    # Normalização dos dados
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)

    # Aplicação do algoritmo KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled)

    # Ajustando os rótulos dos clusters para começar de 1
    df_scaled['cluster'] += 1

    # Exibindo média e desvio padrão por cluster
    df_cluster_desc = df_scaled.groupby('cluster').agg({col: ['mean', 'std'] for col in clustering_cols})
    df_cluster_desc.columns = [f'{col}_{stat}' for col, stat in df_cluster_desc.columns]  # Renomeando as estatísticas para português

    # Exibindo média e desvio padrão
    st.write("<h3>Estatísticas por Cluster:</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col in clustering_cols:
        with c1 if col == clustering_cols[0] else c2:
            st.write(f"<h4>Média e Desvio Padrão de {col}:</h4>", unsafe_allow_html=True)
            st.write(df_cluster_desc[f'{col}_mean'].to_frame('Média').join(df_cluster_desc[f'{col}_std'].to_frame('Desvio Padrão')).to_html(index=True, justify='center', classes='dataframe'), unsafe_allow_html=True)

    # Visualização dos clusters em 2D usando PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled.drop('cluster', axis=1))
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['cluster'] = df_scaled['cluster']

    # Plot dos clusters
    fig = go.Figure()
    for cluster in sorted(df_pca['cluster'].unique()):
        df_cluster = df_pca[df_pca['cluster'] == cluster]
        fig.add_trace(go.Scatter(x=df_cluster['PC1'], y=df_cluster['PC2'], mode='markers', name=f'Cluster {cluster}'))

    fig.update_layout(title='Clusters Visualizados em 2D (KMeans)',
                      xaxis_title='PC1', yaxis_title='PC2')
    st.plotly_chart(fig)

elif algorithm == 'DBSCAN':
    eps = st.slider('Valor de Eps (DBSCAN)', min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    min_samples = st.slider('Número Mínimo de Amostras (DBSCAN)', min_value=2, max_value=20, value=5)

    # Pré-processamento dos dados
    df_processed = df[clustering_cols].dropna()

    # Normalização dos dados
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_processed), columns=df_processed.columns)

    # Aplicação do algoritmo DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_imputed['cluster'] = dbscan.fit_predict(df_imputed)

    # Ajustando os rótulos dos clusters para começar de 1
    cluster_labels = np.unique(df_imputed['cluster'])
    if -1 in cluster_labels:
        cluster_labels = cluster_labels[cluster_labels != -1]
        df_imputed['cluster'] = np.where(df_imputed['cluster'] != -1, df_imputed['cluster'] + 2, 1)
    else:
        df_imputed['cluster'] += 1

    # Exibindo média e desvio padrão por cluster
    df_cluster_desc = df_imputed.groupby('cluster').agg({col: ['mean', 'std'] for col in clustering_cols})
    df_cluster_desc.columns = [f'{col}_{stat}' for col, stat in df_cluster_desc.columns]  # Renomeando as estatísticas para português

    # Exibindo média e desvio padrão
    st.write("<h3>Estatísticas por Cluster:</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col in clustering_cols:
        with c1 if col == clustering_cols[0] else c2:
            st.write(f"<h4>Média e Desvio Padrão de {col}:</h4>", unsafe_allow_html=True)
            st.write(df_cluster_desc[f'{col}_mean'].to_frame('Média').join(df_cluster_desc[f'{col}_std'].to_frame('Desvio Padrão')).to_html(index=True, justify='center', classes='dataframe'), unsafe_allow_html=True)

    # Visualização dos clusters em 2D usando PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_imputed.drop('cluster', axis=1))
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['cluster'] = df_imputed['cluster']

    # Plot dos clusters
    fig = go.Figure()
    for cluster in sorted(df_pca['cluster'].unique()):
        df_cluster = df_pca[df_pca['cluster'] == cluster]
        fig.add_trace(go.Scatter(x=df_cluster['PC1'], y=df_cluster['PC2'], mode='markers', name=f'Cluster {cluster}'))

    fig.update_layout(title='Clusters Visualizados em 2D (DBSCAN)',
                      xaxis_title='PC1', yaxis_title='PC2')
    st.plotly_chart(fig)
