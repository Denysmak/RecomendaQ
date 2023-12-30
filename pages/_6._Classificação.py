import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier  # Adicionado o import para DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

def train_random_forest(X_train_scaled, y_train, X_test_scaled):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return predictions

def train_gradient_boosting(X_train_scaled, y_train, X_test_scaled):
    model = GradientBoostingClassifier(learning_rate=0.2, n_estimators=200)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return predictions

def train_decision_tree(X_train_scaled, y_train, X_test_scaled):
    model = DecisionTreeClassifier()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return predictions
    
def display_metrics(model_name, y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=1)

    st.write(f'Métricas para {model_name}:')
    st.write(f'Acurácia: {accuracy}')
    st.write(f'Precisão: {precision}')
    st.write(f'Revocação: {recall}')
    st.write(f'F1-Score: {f1}')
    st.write('\n')

    # Gráfico de barras para métricas
    metrics_df = pd.DataFrame({
        'Métrica': ['Acurácia', 'Precisão', 'Revocação', 'F1-Score'],
        'Valor': [accuracy, precision, recall, f1]
    })

    st.bar_chart(metrics_df.set_index('Métrica'))

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

    st.sidebar.header("Configurações")
    
    # Removendo colunas desnecessárias
    df = df.drop(columns=['active_years'])
    df = df.drop(columns=['publish_date'])
    df = df.drop(columns=['Rating'])

    # Removendo linhas com valores ausentes
    df.dropna(subset=['final_year', 'Price', 'publish_year', 'publish_month', 'publish_day', 'Rating_Age', 'Imprint', 'Format'], inplace=True)

    # Escolhendo a variável Y
    selected_y = st.sidebar.selectbox("Escolha a coluna alvo", ['Rating_Age', 'Format', 'Imprint'])

    # Excluindo a variável Y da lista de opções para X
    available_x = [col for col in df.columns if col != selected_y]

    # Escolhendo as variáveis X
    selected_x = st.sidebar.multiselect("Escolha as características", available_x)

    if st.sidebar.button("Executar Modelos"):
        st.header("Resultados")

        # Divisão e escalonamento dos dados
        X_train, X_test, y_train, y_test = split_data(df, selected_x, selected_y)
        if X_train is not None:
            # Escalonando os dados
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

            # Treinamento e avaliação do modelo Random Forest
            rf_predictions = train_random_forest(X_train_scaled, y_train, X_test_scaled)
            display_metrics('Random Forest', y_test, rf_predictions)

            # Treinamento e avaliação do modelo Gradient Boosting
            gb_predictions = train_gradient_boosting(X_train_scaled, y_train, X_test_scaled)
            display_metrics('Gradient Boosting', y_test, gb_predictions)

            #treinamento e avaliação do modelo decision tree classifier
            dt_predictions = train_decision_tree(X_train_scaled, y_train, X_test_scaled)
            display_metrics('Decision Tree Classifier', y_test, dt_predictions)


if __name__ == '__main__':
    main()
