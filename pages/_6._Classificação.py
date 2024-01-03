import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  

def load_data(dataset_path):
    dados = pd.read_parquet(dataset_path)
    return dados.copy()

def preprocess_data(df):
    # Transformando os valores da coluna publish_date em data numérica e separando em dia, mês e ano
    df['publish_date'] = df['publish_date'].apply(lambda x: pd.to_datetime(x, format='%B %d, %Y', errors='coerce') if pd.notnull(x) and x != 'None' else x)
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df['publish_date'] = df['publish_date'].dt.normalize()
    df['publish_year'] = df['publish_date'].dt.year
    df['publish_month'] = df['publish_date'].dt.month
    df['publish_day'] = df['publish_date'].dt.day

    # Convertendo a coluna 'active_years' para conter apenas o último ano
    df['final_year'] = df['active_years'].str.extract('(\d{4})\D*(\d{4})?').astype(float).iloc[:, -1]

    # Substituindo os valores "Free" na coluna 'Price' por 0.00
    df['Price'] = df['Price'].replace(' Free', '0.00')
    df['Price'] = df['Price'].replace('None', '0.00').replace('[\$,]', '', regex=True).astype(float)
    return df

def map_ratings(df):
    rating_age_mapping = {
        ' Rated T+': 13,
        ' None': 1,
        ' Rated T': 13,
        ' ALL AGES': 0,
        ' A': 0,
        ' Parental Advisory': 16,
        ' Marvel Psr': 13,
        ' No Rating': 1,
        ' MARVEL PSR': 13,
        ' T': 13,
        ' RATED T': 13,
        ' Max': 17,
        ' RATED T+': 13,
        ' RATED A': 0,
        ' All Ages': 0,
        ' T+': 13,
        ' Rated a': 0,
        ' Rated A': 0,
        ' Parental Advisory/Explicit Content': 16,
        ' PARENTAL SUPERVISION': 16,
        ' PARENTAL ADVISORY': 16,
        ' Mature': 17,
        ' MARVEL PSR+': 13,
        ' EXPLICIT CONTENT': 17,
        ' PARENTAL ADVISORYSLC': 16,
        ' Parental AdvisorySLC': 16,
        ' Parental Advisoryslc': 16,
        ' Explicit Content': 17,
        ' PARENTAL ADVISORY/EXPLICIT CONTENT': 17,
        ' NO RATING': 1,
        ' NOT IN ORACLE': 1,
        ' Parental Guidance': 13,
        ' Ages 10 & Up': 10,
        ' Not in Oracle': 1,
        ' MAX': 17,
        ' Marvel Psr+': 13,
        ' Ages 9+': 9,
    }

    df['Rating_Age'] = df['Rating'].map(rating_age_mapping)
    return df

def map_formats(df):
    format_mapping = {
        ' Infinite Comic': 0,
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
        ' Graphic Novel': 11
    }

    df['Format'] = df['Format'].map(format_mapping)
    df['Format'].fillna(-1, inplace=True)
    return df

def map_imprints(df):
    imprint_mapping = {
        ' Marvel Universe': 0,
        ' None': 1,
        ' MARVEL UNIVERSE': 0,
        ' Max': 2,
        ' Ultimate Universe': 3,
        ' Marvel Knights': 4,
        ' MARVEL KNIGHTS': 4,
        ' Atlas [20th Century Comics Corp]': 5,
        ' MARVEL ARCHIVE': 6,
        ' Custom': 7,
        ' Licensed Publishing': 8,
        ' Icon': 9,
        ' Timely / Atlas [Timely Publications': 10,
        ' ICON': 9,
        ' New Universe': 11,
        ' LICENSED PUBLISHING': 8,
        ' MAX': 2,
        ' Marvel Adventures': 12,
        ' Licenced Publishing': 8,
        ' MARKETING': 13,
        ' MARVEL ADVENTURES': 12,
        ' DABEL BROTHERS': 14,
        ' Alterniverse': 15,
        ' marvel age': 16,
        ' Marvel Archive': 6,
        ' Atlas [Medalion Publishing Corp.]': 17,
        ' MARVEL ILLUSTRATED': 18,
        ' Marvel Illustrated': 18,
        ' OUTREACH/NEW READER': 19,
        ' ESSENTIALS': 20,
        ' LICENCED PUBLISHING': 8,
        ' ULTIMATE': 3,
        ' 2099': 21,
        ' Not in Oracle': 22,
        ' MARVEL UNIVERS': 0,
        ' Atlas': 23,
        ' Outreach/New Reader': 19,
        ' ULTIMATE UNIVERSE': 3,
        ' NOT IN ORACLE': 22
    }
    
    df['Imprint'] = df['Imprint'].map(imprint_mapping)
    df['Imprint'].fillna(-1, inplace=True)
    return df

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), feature_names[indices])
    plt.tight_layout()
    st.pyplot(plt)

def split_data(df, selected_x, selected_y):
    selected_columns = selected_x + [selected_y]
    
    if not all(col in df.columns for col in selected_columns):
        st.error("Colunas selecionadas não estão presentes no DataFrame.")
        return None, None, None, None
    
    X = df[selected_x]
    y = df[selected_y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def train_random_forest(X_train, X_train_scaled, y_train, X_test_scaled, n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return model, predictions

def train_gradient_boosting(X_train, X_train_scaled, y_train, X_test_scaled, n_estimators=200, learning_rate=0.2, max_depth=3, min_samples_split=2, min_samples_leaf=1):
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return model, predictions

def train_decision_tree(X_train, X_train_scaled, y_train, X_test_scaled, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return model, predictions
    
def display_metrics(model_name, y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=1)

    st.write(f'Métricas para {model_name}:')
    st.table(pd.DataFrame({
        'Métrica': ['Acurácia', 'Precisão', 'Revocação', 'F1-Score'],
        'Valor': [accuracy, precision, recall, f1]
    }))
    st.write('\n')

def main():
    st.title("Classificação com Modelos de Machine Learning")

    dataset_path = 'data/Marvel_Comics.parquet'
    
    # Carregando os dados
    dados = load_data(dataset_path)
    df = preprocess_data(dados[['active_years', 'Price', 'Rating', 'publish_date', 'Format', 'Imprint']].copy())

    # Mapeamento e tratamento dos dados
    df = map_ratings(df)
    df = map_formats(df)
    df = map_imprints(df)

    # Removendo colunas desnecessárias
    df = df.drop(columns=['active_years'])
    df = df.drop(columns=['publish_date'])
    df = df.drop(columns=['Rating'])

    # Removendo linhas com valores ausentes
    df.dropna(subset=['final_year', 'Price', 'publish_year', 'publish_month', 'publish_day', 'Rating_Age', 'Imprint', 'Format'], inplace=True)

    st.header("Configurações")
    
    # Escolhendo a variável Y
    selected_y = st.selectbox("Escolha a coluna alvo", ['Rating_Age', 'Format', 'Imprint'])

    # Excluindo a variável Y da lista de opções para X
    available_x = [col for col in df.columns if col != selected_y]

    # Escolhendo as variáveis X
    selected_x = st.multiselect("Escolha as características", available_x)

    # Adicione controles deslizantes para ajustar os hiperparâmetros
    n_estimators = st.slider("Número de Estimadores (Random Forest e Gradient Boosting)", 10, 500, 200, 10)
    max_depth = st.slider("Profundidade Máxima (Random Forest e Gradient Boosting)", 1, 20, 10, 1)
    min_samples_split = st.slider("Número Mínimo de Amostras para Dividir (Random Forest e Gradient Boosting)", 2, 20, 2, 1)
    min_samples_leaf = st.slider("Número Mínimo de Amostras em Folha (Random Forest e Gradient Boosting)", 1, 20, 1, 1)
    learning_rate = st.slider("Taxa de Aprendizado (Gradient Boosting)", 0.01, 1.0, 0.2, 0.01)
    max_depth_dt = st.slider("Profundidade Máxima (Decision Tree Classifier)", 1, 20, 10, 1)
    min_samples_split_dt = st.slider("Número Mínimo de Amostras para Dividir (Decision Tree Classifier)", 2, 20, 2, 1)
    min_samples_leaf_dt = st.slider("Número Mínimo de Amostras em Folha (Decision Tree Classifier)", 1, 20, 1, 1)

    if st.button("Executar Modelos"):
        st.header("Resultados")

        # Divisão e escalonamento dos dados
        X_train, X_test, y_train, y_test = split_data(df, selected_x, selected_y)
        if X_train is not None:
            # Escalonando os dados
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

            # Treinamento do modelo Random Forest
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            rf_model.fit(X_train_scaled, y_train)

            # Avaliação do modelo Random Forest
            rf_predictions = rf_model.predict(X_test_scaled)

            st.subheader("Random Forest")
            display_metrics('Random Forest', y_test, rf_predictions)

            # Adiciona o gráfico de feature importance após a tabela de métricas para o Random Forest
            st.subheader("Feature Importance - Random Forest")
            plot_feature_importance(rf_model, X_train.columns)

            # Treinamento do modelo Gradient Boosting
            gb_model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            gb_model.fit(X_train_scaled, y_train)

            # Avaliação do modelo Gradient Boosting
            gb_predictions = gb_model.predict(X_test_scaled)

            st.subheader("Gradient Boosting")
            display_metrics('Gradient Boosting', y_test, gb_predictions)

            # Adiciona o gráfico de feature importance após a tabela de métricas para o Gradient Boosting
            st.subheader("Feature Importance - Gradient Boosting")
            plot_feature_importance(gb_model, X_train.columns)

            # Treinamento do modelo Decision Tree Classifier
            dt_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            dt_model.fit(X_train_scaled, y_train)

            # Avaliação do modelo Decision Tree Classifier
            dt_predictions = dt_model.predict(X_test_scaled)

            st.subheader("Decision Tree Classifier")
            display_metrics('Decision Tree Classifier', y_test, dt_predictions)

            # Adiciona o gráfico de feature importance após a tabela de métricas para o Decision Tree Classifier
            st.subheader("Feature Importance - Decision Tree Classifier")
            plot_feature_importance(dt_model, X_train.columns)

            # Comparação dos modelos com tabela
            st.subheader("Comparação dos Modelos")
            metrics_df = pd.DataFrame({
                'Métrica': ['Acurácia', 'Precisão', 'Revocação', 'F1-Score'],
                'Random Forest': [accuracy_score(y_test, rf_predictions), precision_score(y_test, rf_predictions, average='weighted', zero_division=1),
                                    recall_score(y_test, rf_predictions, average='weighted', zero_division=1), f1_score(y_test, rf_predictions, average='weighted', zero_division=1)],
                'Gradient Boosting': [accuracy_score(y_test, gb_predictions), precision_score(y_test, gb_predictions, average='weighted', zero_division=1),
                                        recall_score(y_test, gb_predictions, average='weighted', zero_division=1), f1_score(y_test, gb_predictions, average='weighted', zero_division=1)],
                'Decision Tree Classifier': [accuracy_score(y_test, dt_predictions), precision_score(y_test, dt_predictions, average='weighted', zero_division=1),
                                            recall_score(y_test, dt_predictions, average='weighted', zero_division=1), f1_score(y_test, dt_predictions, average='weighted', zero_division=1)]
            })

            st.table(metrics_df.set_index('Métrica'))


if __name__ == '__main__':
    main()
