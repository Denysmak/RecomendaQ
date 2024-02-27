import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from datetime import datetime

dataset = 'data/Marvel_Comics.parquet'
df = pd.read_parquet(dataset)

df['Price'] = df['Price'].str.replace('Free', '0.00')
df['Price'] = df['Price'].str.replace('$', '').astype(float)

def adjust_price_for_inflation(original_price, original_year, current_year=None, inflation_rate=0.034):
    if current_year is None:
        current_year = datetime.now().year
    years_passed = current_year - original_year
    inflation_multiplier = (1 + inflation_rate) ** years_passed
    adjusted_price = original_price * inflation_multiplier
    return adjusted_price

# Limpeza da coluna 'active_years' para obter os anos de início e término das séries
df['start_year'] = df['active_years'].str.extract(r'(\d{4})').astype(float)
df['start_year'].fillna(np.nan, inplace=True)
df.loc[df['start_year'] < 1800, 'start_year'] = np.nan

# Ajuste de preços para inflação
df['Price'] = df.apply(lambda row: adjust_price_for_inflation(row['Price'], row['start_year']), axis=1)

# Tratando as inconsistências na coluna de Rating
df['Rating'] = df['Rating'].str.lower().str.replace('rated', '').str.strip()
df['Rating'].replace(['no rating', None], 'não classificados', inplace=True)

# Filtrando as 8 primeiras classificações de idade (ratings)
top_ratings = df['Rating'].value_counts().index[:8]
df_filtered = df[df['Rating'].isin(top_ratings)]
df_filtered_with_unrated = df[df['Rating'].isin(top_ratings)]

# Dicionário de mapeamento das classificações
rating_mapping = {
    't+': 13,
    't': 10,
    'parental advisory': 15,
    'all ages': 0,
    'marvel psr': 15,
    'a': 9,
    'explicit content': 18
}

# Substituir os valores na coluna 'Rating' usando o dicionário de mapeamento
df['Rating'] = df['Rating'].map(rating_mapping)

df.loc[df['Rating'] == 'não classificados', 'Rating'] = np.nan

color_scale = ['#00ccff','#cc00ff','#ffcc00','#0066bb','#6600bb','#bb0066','#bb6600','#ff0066','#66ff66','#ee0503']
n_clusters = 3
clustering_cols_opts = ['Price','Rating','start_year','Format']
clustering_cols = clustering_cols_opts.copy()

def create_dfs():
    clustering_cols_opts = ['Price','Rating','start_year','Format']
    df_raw = create_df_raw(clustering_cols_opts)
    df_clean = df_raw.dropna()
    df_clusters = create_df_clusters(df)
    return {
        'df_raw': (df_raw, 'Dataframe Original', 'Dataframe original do Titanic, com um subconjunto de colunas utilizados para o agrupamento (clusterização).'),
        'df_clean': (df_clean, 'Dataframe Sem Nulos', 'Dataframe após a remoção dos registros que possuíam valores nulos nas colunas selecionadas.'),
        'df_clusters': (df_clusters, 'Clusterização do Dataframe Codificado e com Normalizações', 'Dataframe após a execução de diferentes normalizações'),
    }
    
def create_df_raw(cols):
    df_raw = df[cols].copy()
    return df_raw

def create_df_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna() 
    df_clusters = df_clean.copy()
    df_clusters['cluster'] = clusterize(df_clean)
    df_clusters['cluster_standard'] = clusterize(df_clean, StandardScaler())
    df_clusters['cluster_minmax'] = clusterize(df_clean, MinMaxScaler())  # Adicionando a coluna cluster_minmax
    return df_clusters



def clusterize(df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    df_result = df[clustering_cols].copy()
    if scaler is not None:
        df_result = scale(df_result, scaler)
    X = df_result.values

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=4294967295)
    return kmeans.fit_predict(X)

def scale(df:pd.DataFrame, scaler):
    scaling_cols = [x for x in ['idade','tarifa'] if x in clustering_cols]
    for c in scaling_cols:
        vals = df[[c]].values
        df[c] = scaler.fit_transform(vals)
    return df

def plot_dataframe(df, title, desc):
    with st.expander(title):
        st.write(f'<i>{desc}</i>', unsafe_allow_html=True)
        c1, _, c2 = st.columns([.5,.05,.45])
        c1.write('<h3>Dados*</h3>', unsafe_allow_html=True)
        c1.dataframe(df, use_container_width=True)
        c2.write('<h3>Descrição</h3>', unsafe_allow_html=True)
        c2.dataframe(df.describe())

def plot_cluster(df: pd.DataFrame, cluster_col: str, cluster_name: str):
    df_cluster_desc = df.groupby(cluster_col).agg({col: ['mean', 'std'] for col in clustering_cols})
    df_cluster_desc.columns = [f'{col}_{stat}' for col, stat in df_cluster_desc.columns]  # Renomeando as estatísticas para português

    # Renomear os clusters para começar de 1
    df_cluster_desc.index = df_cluster_desc.index + 1

    # Exibindo média e desvio padrão
    st.write("<h3>Estatísticas por Cluster:</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col in clustering_cols:
        with c1 if col == clustering_cols[0] else c2:
            st.write(f"<h4>Média e Desvio Padrão de {col}:</h4>", unsafe_allow_html=True)
            st.write(df_cluster_desc[f'{col}_mean'].to_frame('Média').join(df_cluster_desc[f'{col}_std'].to_frame('Desvio Padrão')).to_html(index=True, justify='center', classes='dataframe'), unsafe_allow_html=True)

    # Gerando os gráficos
    custom_colors = ['#00ccff', '#cc00ff', '#ffcc00', '#0066bb', '#6600bb', '#bb0066', '#bb6600', '#ff0066', '#66ff66', '#ee0503']
    cluster_labels = sorted(df[cluster_col].unique())  # Garante que os clusters estejam em ordem crescente
    num_clusters = len(cluster_labels)
    custom_colors_cluster = custom_colors[:num_clusters]
    cluster_colors = {label: color for label, color in zip(cluster_labels, custom_colors_cluster)}
    fig = go.Figure()
    for label, color in cluster_colors.items():
        if color == '#FFFFFF' or color == '#000000':  
            color = '#FF5733'  
        cluster_data = df[df[cluster_col] == label]
        fig.add_trace(go.Scatter(
            x=cluster_data[clustering_cols[0]],
            y=cluster_data[clustering_cols[1]],
            mode='markers',
            marker=dict(color=color),
            name=f'Cluster {label + 1}'  # Inicia a numeração dos clusters a partir de 1
        ))
    fig.update_layout(
        xaxis_title=clustering_cols[0],
        yaxis_title=clustering_cols[1],
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)  

def build_header():
    st.write('<h1>Agrupamento (<i>Clustering</i>) com a base do Marvel_Comics</h1>', unsafe_allow_html=True)

def build_body_kmeans(key):
    global n_clusters, clustering_cols
    c1, c2 = st.columns(2)
    clustering_cols = c1.multiselect(f'Colunas - KMeans {key}', options=clustering_cols_opts, default=clustering_cols_opts[0:2])
    if len(clustering_cols) < 2:
        st.error('É preciso selecionar pelo menos 2 colunas.')
        return
    n_clusters = c2.slider(f'Quantidade de Clusters (KMeans) {key}', min_value=2, max_value=10, value=3)

    dfs = create_dfs()
    for df, title, desc in dfs.values():
        plot_dataframe(df, title, desc)
    df_clusters = dfs['df_clusters'][0]

    # Opção para selecionar a clusterização desejada
    cluster_option = st.selectbox(f'Selecione a opção de clusterização (KMeans) {key}', [
        'Cluster Sem Normalização',
        'Cluster Com Normalização Padrão',
        'Cluster Com Normalização MinMax'
    ])

    if cluster_option == 'Cluster Sem Normalização':
        plot_cluster(df_clusters, 'cluster', 'Cluster Sem Normalização (KMeans) kmeans')
    elif cluster_option == 'Cluster Com Normalização Padrão':
        plot_cluster(df_clusters, 'cluster_standard', 'Cluster Com Normalização Padrão (KMeans) kmeans')
    elif cluster_option == 'Cluster Com Normalização MinMax':
        plot_cluster(df_clusters, 'cluster_minmax', 'Cluster Com Normalização MinMax (KMeans) kmeans')


def build_body_dbscan(key):
    global clustering_cols
    c1, c2 = st.columns(2)
    clustering_cols = c1.multiselect(f'Colunas - DBSCAN {key}', options=clustering_cols_opts, default=clustering_cols_opts[0:2])
    if len(clustering_cols) < 2:
        st.error('É preciso selecionar pelo menos 2 colunas.')
        return

    eps_value = c2.slider(f'Valor de Eps (DBSCAN) {key}', min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    min_samples_value = st.slider(f'Número Mínimo de Amostras (DBSCAN) {key}', min_value=2, max_value=20, value=5)

    selected_columns = df[clustering_cols]
    imputer = SimpleImputer(strategy='median')
    selected_columns_imputed = pd.DataFrame(imputer.fit_transform(selected_columns), columns=clustering_cols)

    dbscan_marvel = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    dbscan_marvel.fit(selected_columns_imputed)
    labels = dbscan_marvel.labels_

    # Atribuir rótulos começando de 1 e garantir que sejam contínuos
    label_counter = 1
    label_map = {}
    for label in labels:
        if label not in label_map:
            label_map[label] = label_counter
            label_counter += 1
    normalized_labels = [label_map[label] for label in labels]

    selected_columns_imputed['Cluster'] = normalized_labels
    
    custom_colors = ['#0305BF', '#00BF63', '#F13638', '#FA00FF', '#FFDE00', '#9E00FF', '#82286B',
                     '#398270', '#C6D3BE', '#F63B63', '#B3431A', '#FEF0E8', '#F857C9', '#9C2745', '#A1D807',
                     '#3EB665', '#72D1CD', '#D42156', '#672403']
    
    cluster_labels = np.unique(normalized_labels)
    num_clusters = len(cluster_labels)
    custom_colors_cluster = custom_colors[:num_clusters]
    cluster_colors = {label: color for label, color in zip(cluster_labels, custom_colors_cluster)}

    fig = go.Figure()
    for label, color in cluster_colors.items():
        cluster_data = selected_columns_imputed[selected_columns_imputed['Cluster'] == label]
        fig.add_trace(go.Scatter(
            x=cluster_data[clustering_cols[0]],
            y=cluster_data[clustering_cols[1]],
            mode='markers',
            marker=dict(color=color),
            name=f'Cluster {label}'
        ))

    fig.update_layout(
        xaxis_title=clustering_cols[0],
        yaxis_title=clustering_cols[1],
        showlegend=True
    )

    with st.expander(f'Clusterização (DBSCAN) {key}'):
        st.plotly_chart(fig)


def build_page():
    build_header()
    build_tabs()


def build_tabs():
    tabs = st.tabs(["KMeans", "DBSCAN"])
    with tabs[0]:
        build_body_kmeans('kmeans')
    with tabs[1]:
        build_body_dbscan('dbscan')


build_page()
