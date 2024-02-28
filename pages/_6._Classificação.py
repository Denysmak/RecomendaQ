import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def load_data(dataset_path):
    dados = pd.read_parquet(dataset_path)
    return dados.copy()

def preprocess_data(df):
    # Transformando os valores da coluna publish_date em data numérica e separando em dia, mês e ano
    df['publish_date'] = df['publish_date'].apply(lambda x: pd.to_datetime(
        x, format='%B %d, %Y', errors='coerce') if pd.notnull(x) and x != 'None' else x)
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df['publish_date'] = df['publish_date'].dt.normalize()
    df['publish_year'] = df['publish_date'].dt.year
    df['publish_month'] = df['publish_date'].dt.month
    df['publish_day'] = df['publish_date'].dt.day

    # Convertendo a coluna 'active_years' para conter apenas o último ano
    df['final_year'] = df['active_years'].str.extract(
        '(\d{4})\D*(\d{4})?').astype(float).iloc[:, -1]

    # Substituindo os valores "Free" na coluna 'Price' por 0.00
    df['Price'] = df['Price'].replace(' Free', '0.00')
    df['Price'] = df['Price'].replace('None', '0.00').replace(
        '[\$,]', '', regex=True).astype(float)

    # Ajustar os preços para a inflação usando a função criada
    current_year = 2023  # Ano atual para o qual queremos ajustar a inflação
    df['Price'] = df.apply(lambda row: adjust_price_for_inflation(
        row['Price'], row['publish_date'], current_year), axis=1)

    df['Rating_Age'] = df['Rating']

    # Retirando todos os nomes após a vírgula nas colunas penciler, writer e cover_artist
    if 'penciler' in df.columns:
        df['penciler'] = df['penciler'].str.split(',').str.get(0)

    if 'writer' in df.columns:
        df['writer'] = df['writer'].str.split(',').str.get(0)

    if 'cover_artist' in df.columns:
        df['cover_artist'] = df['cover_artist'].str.split(',').str.get(0)

    # Remova todos os valores 'NaN' em todas as colunas
    df = df.dropna()

    # Remova todos os valores 'None' em todas as colunas
    for coluna in df.columns:
        df = df[(df[coluna] != 'None')]
    return df

def process_artists(data):
    if isinstance(data, pd.DataFrame):
        all_artists = set()  # Conjunto vazio para armazenar todos os artistas únicos

        # Verificar a presença de cada coluna e adicionar os artistas únicos ao conjunto
        if 'penciler' in data.columns:
            all_artists |= set(data['penciler'])
        if 'writer' in data.columns:
            all_artists |= set(data['writer'])
        if 'cover_artist' in data.columns:
            all_artists |= set(data['cover_artist'])

        # Criar um mapa único para todos esses artistas
        artist_map = {artist: f"{i:02d}" for i, artist in enumerate(sorted(all_artists))}

        # Aplicar o mapa a cada uma das colunas relevantes
        if 'penciler' in data.columns:
            data['penciler'] = data['penciler'].map(artist_map)
        if 'writer' in data.columns:
            data['writer'] = data['writer'].map(artist_map)
        if 'cover_artist' in data.columns:
            data['cover_artist'] = data['cover_artist'].map(artist_map)

        return data
    elif isinstance(data, dict):
        all_artists = set()  # Conjunto vazio para armazenar todos os artistas únicos

        # Verificar a presença de cada chave e adicionar os artistas únicos ao conjunto
        if 'penciler' in data:
            all_artists |= set(data['penciler'])
        if 'writer' in data:
            all_artists |= set(data['writer'])
        if 'cover_artist' in data:
            all_artists |= set(data['cover_artist'])

        # Criar um mapa único para todos esses artistas
        artist_map = {artist: f"{i:02d}" for i, artist in enumerate(sorted(all_artists))}

        # Aplicar o mapa a cada uma das chaves relevantes
        if 'penciler' in data:
            data['penciler'] = [artist_map[artist] for artist in data['penciler']]
        if 'writer' in data:
            data['writer'] = [artist_map[artist] for artist in data['writer']]
        if 'cover_artist' in data:
            data['cover_artist'] = [artist_map[artist] for artist in data['cover_artist']]

        return data
    else:
        raise ValueError(
            "Os dados de entrada devem ser um DataFrame ou um dicionário.")

def preprocess_user_input(user_input):
    # Transforming the values of the publish_date column into numeric date and splitting into day, month, and year
    if 'publish_date' in user_input:
        user_input['publish_date'] = pd.to_datetime(
            user_input['publish_date'], format='%B %d, %Y', errors='coerce').normalize()
        user_input['publish_year'] = user_input['publish_date'].dt.year
        user_input['publish_month'] = user_input['publish_date'].dt.month
        user_input['publish_day'] = user_input['publish_date'].dt.day
        del user_input['publish_date']

    # Mapping artists and other features
    user_input = process_artists(user_input)
    
    # Ensuring user_input is a DataFrame
    user_input_df = pd.DataFrame(user_input)

    # Mapping the Format column, if present
    if 'Format' in user_input_df:
        user_input_df['Format'], format_mapping, format_mapping_inverse = map_formats(user_input_df[['Format']])
    
    # Mapping the Imprint column, if present
    if 'Imprint' in user_input_df:
        user_input_df['Imprint'], imprint_mapping, imprint_mapping_inverse = map_imprints(user_input_df[['Imprint']])

    return user_input_df


def map_ratings(df):
    rating_age_mapping = {
        ' Rated T+': 13,
        ' None': -1,
        ' Rated T': 13,
        ' ALL AGES': 1,
        ' A': 1,
        ' Parental Advisory': 16,
        ' Marvel Psr': 15,
        ' No Rating': 1,
        ' MARVEL PSR': 15,
        ' T': 13,
        ' RATED T': 13,
        ' Max': 17,
        ' RATED T+': 13,
        ' RATED A': 1,
        ' All Ages': 1,
        ' T+': 13,
        ' Rated a': 1,
        ' Rated A': 1,
        ' Parental Advisory/Explicit Content': 16,
        ' PARENTAL SUPERVISION': 15,
        ' PARENTAL ADVISORY': 16,
        ' Mature': 17,
        ' MARVEL PSR+': 15,
        ' EXPLICIT CONTENT': 17,
        ' PARENTAL ADVISORYSLC': 16,
        ' Parental AdvisorySLC': 16,
        ' Parental Advisoryslc': 16,
        ' Explicit Content': 17,
        ' PARENTAL ADVISORY/EXPLICIT CONTENT': 17,
        ' NO RATING': 1,
        ' NOT IN ORACLE': -1,
        ' Parental Guidance': 13,
        ' Ages 10 & Up': 10,
        ' Not in Oracle': -1,
        ' MAX': 17,
        ' Marvel Psr+': 15,
        ' Ages 9+': 9,
    }
    
    # Criar mapeamento inverso
    rating_age_mapping_inverse = {v: k for k, v in rating_age_mapping.items()}

    df['Rating_Age'] = df['Rating_Age'].map(rating_age_mapping)
    return df, rating_age_mapping, rating_age_mapping_inverse

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
    
    # Criar mapeamento inverso
    format_mapping_inverse = {v: k for k, v in format_mapping.items()}

    df['Format'] = df['Format'].map(format_mapping)
    df['Format'].fillna(-1, inplace=True)
    return df, format_mapping, format_mapping_inverse

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
    
    # Criar mapeamento inverso
    imprint_mapping_inverse = {v: k for k, v in imprint_mapping.items()}

    df['Imprint'] = df['Imprint'].map(imprint_mapping)
    df['Imprint'].fillna(-1, inplace=True)
    
    return df, imprint_mapping, imprint_mapping_inverse

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), feature_names[indices])
    plt.tight_layout()
    st.pyplot(plt)


def split_user_data(df, selected_x, selected_y, user_inputs):
    selected_columns = selected_x + [selected_y]

    if not all(col in df.columns for col in selected_columns):
        st.error("Colunas selecionadas não estão presentes no DataFrame.")
        return None, None, None, None

    X_train = df[selected_x]
    y_train = df[selected_y]
    
    if not all(col in user_inputs for col in selected_columns):
        st.error("Colunas selecionadas não estão presentes nos dados de entrada do usuário.")
        return None, None, None, None
    
    X_test = user_inputs[selected_x]
    y_test = user_inputs[selected_y]

    # Utilizando RandomOverSampler para balancear as classes nos dados de treino
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # Utilizando RandomUnderSampler para balancear as classes nos dados de treino
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)

    return X_train_resampled, X_test, y_train_resampled, y_test

def split_data(df, selected_x, selected_y):
    selected_columns = selected_x + [selected_y]
    
    if not all(col in df.columns for col in selected_columns):
        st.error("Colunas selecionadas não estão presentes no DataFrame.")
        return None, None, None, None
    
    X = df[selected_x]
    y = df[selected_y]

    # Utilizando RandomOverSampler para balancear as classes nos dados de treino
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Utilizando RandomUnderSampler para balancear as classes nos dados de treino
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def train_random_forest(X_train_scaled, y_train, X_test_scaled, n_estimators=None, max_depth=None, min_samples_split=None, min_samples_leaf=None):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return model, predictions

def train_gradient_boosting(X_train_scaled, y_train, X_test_scaled, n_estimators=200, learning_rate=0.2, max_depth=3, min_samples_split=2, min_samples_leaf=1):
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                       max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return model, predictions

def train_decision_tree(X_train_scaled, y_train, X_test_scaled, max_depth=None, min_samples_split=None, min_samples_leaf=None):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return model, predictions


def display_metrics(model_name, y_test, predictions_test, class_labels, inverse_mapping):
    # Convertendo os rótulos de y_test para o mesmo tipo de predictions_test
    y_test = y_test.astype(predictions_test.dtype)

    accuracy = accuracy_score(y_test, predictions_test)
    precision = precision_score(
        y_test, predictions_test, average='weighted', zero_division=1)
    recall = recall_score(y_test, predictions_test,
                          average='weighted', zero_division=1)
    f1 = f1_score(y_test, predictions_test,
                  average='weighted', zero_division=1)

    st.write(f'Métricas para {model_name} com base no conjunto de teste:')
    st.table(pd.DataFrame({
        'Métrica': ['Acurácia', 'Precisão', 'Revocação', 'F1-Score'],
        'Valor': [accuracy, precision, recall, f1]
    }))
    st.write('\n')
    # Mapear as previsões de volta para os nomes correspondentes
    inverse_predictions = [inverse_mapping.get(label, 'Classe Desconhecida') 
                           if label != -1.0 else 'Classe Desconhecida' for label in predictions_test]
    
    # Matriz de Confusão
    st.subheader('Matriz de Confusão:')
    cm = confusion_matrix(y_test, predictions_test)
    
    # Garantir que todos os valores presentes nas previsões estejam no mapeamento inverso
    all_labels = set(predictions_test).union(set(y_test))
    labels = [label for label in all_labels if label in inverse_mapping]

    # Adiciona todas as classes representadas no conjunto de teste
    all_classes = np.unique(np.concatenate((y_test, predictions_test)))
    all_classes.sort()

    # Exibir a matriz de confusão em formato de tabela
    conf_matrix_df = pd.DataFrame(
        data=cm, index=np.unique(np.concatenate([y_test, predictions_test])), columns=np.unique(np.concatenate([y_test, predictions_test])))
    st.table(conf_matrix_df)

def display_majority_percentage(y_test, predictions_test, inverse_mapping):
    # Mapear as previs천es de volta para os nomes correspondentes
    inverse_predictions = [inverse_mapping.get(label, "Classe Desconhecida") for label in predictions_test]

    # Criar DataFrame com as previs천es mapeadas
    predictions_df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': predictions_test,
        'Mapped Prediction': [inverse_mapping.get(label, 'Classe Desconhecida') if label != -1.0 else 'Valor Especial' for label in predictions_test]
    })

    # Calcular a porcentagem de cada classe nas previs천es
    class_percentages = predictions_df['Mapped Prediction'].value_counts(normalize=True) * 100

    # Encontrar a classe com a maior porcentagem
    majority_class = class_percentages.idxmax()
    majority_percentage = class_percentages.max()

    st.subheader('Previsões Mapeadas com Porcentagem da Maioria:')
    st.table(pd.DataFrame({
        'Porcentagem': class_percentages.apply(lambda x: '{:.2f}%'.format(x))
    }))

def adjust_price_for_inflation(original_price, original_year, current_year, inflation_rate=0.034):
    years_passed = current_year - original_year.year  # Extrai o ano da data
    inflation_multiplier = (1 + inflation_rate) ** years_passed
    adjusted_price = original_price * inflation_multiplier
    return round(adjusted_price, 2)  # Round para 2 casas decimais

def main():
    st.title("Classificação com Modelos de Machine Learning")

    dataset_path = 'data/Marvel_Comics.parquet'

    np.random.seed(42)

    # Carregando os dados
    dados = load_data(dataset_path)
    df = preprocess_data(dados[['active_years', 'Price', 'Rating', 'publish_date', 'Format', 'Imprint', 'cover_artist', 'writer', 'penciler']].copy())

    df, rating_age_mapping, rating_age_mapping_inverse = map_ratings(df)
    
    # Removendo colunas desnecessárias
    df = df.drop(columns=['active_years'])
    df = df.drop(columns=['publish_date'])
    df = df.drop(columns=['Rating'])

    # Removendo linhas com valores ausentes
    df.dropna(subset=['final_year', 'Price', 'publish_year', 'publish_month', 'publish_day',
                      'Rating_Age', 'Imprint', 'Format', 'cover_artist', 'writer', 'penciler'], inplace=True)
    
    st.header("Configurações")

    selected_y = st.selectbox("Escolha a coluna alvo", 
                              ['Rating_Age', 'Format', 'Imprint'])
    
    # Excluindo a variável Y da lista de opções para X
    available_x = [col for col in df.columns if col != selected_y]

    # Escolhendo as variáveis X
    selected_x = st.multiselect("Escolha as características", available_x)
    
    tela = st.selectbox('Escolher os Dados?', ('Sim', 'Não'))
    
    if tela == 'Sim':
        user_input = {}

        user_input[selected_y] = st.selectbox(f'Escolha o valor para {selected_y}:', df[selected_y].unique())

        for feature in selected_x:
            user_input[feature] = st.selectbox(f'Escolha o valor para {feature}:', df[feature].unique())
        
        user_input = pd.DataFrame(user_input, index=[0])

        # Ajuste de HyperParâmetros
        n_estimators = st.slider("Número de Estimadores (Random Forest e Gradient Boosting)", 10, 500, 200, 10)
        max_depth = st.slider("Profundidade Máxima (Random Forest, Decision Tree Classifier e Gradient Boosting)", 1, 20, 10, 1)
        min_samples_split = st.slider("Número Mínimo de Amostras para Dividir (Random Forest, Decision Tree Classifier e Gradient Boosting)", 2, 20, 2, 1)
        min_samples_leaf = st.slider("Número Mínimo de Amostras em Folha (Random Forest, Decision Tree Classifier e Gradient Boosting)", 1, 20, 1, 1)
        learning_rate = st.slider("Taxa de Aprendizado (Gradient Boosting)", 0.01, 1.0, 0.2, 0.01)

        if st.button("Executar Modelos"):
            st.header("Treinamento e Avaliação do Modelo")

            # Mapeamento e tratamento dos dados
            df, format_mapping, format_mapping_inverse = map_formats(df)
            df, imprint_mapping, imprint_mapping_inverse = map_imprints(df)
            df = process_artists(df)
            
            # Mapeamentos inversos correspondentes às colunas alvo
            inverse_mapping_dict = {
                'Rating_Age': rating_age_mapping_inverse,
                'Format': format_mapping_inverse,
                'Imprint': imprint_mapping_inverse
            }
            
            # Obtendo o mapeamento inverso correspondente à coluna alvo escolhida
            inverse_mapping = inverse_mapping_dict[selected_y]

            # Pré-processamento dos dados de entrada do usuário
            user_input = preprocess_user_input(user_input)

            # Removendo linhas com valores ausentes
            df.dropna(subset=['final_year', 'Price', 'publish_year', 'publish_month', 'publish_day', 'Rating_Age', 'Imprint', 'Format', 'cover_artist', 'writer', 'penciler'], inplace=True)

            X_train, X_test, y_train, y_test = split_user_data(df, selected_x, selected_y, user_input)
            
            if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
                X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
                
                # Contagem de classes na coluna selecionada
                class_counts = df[selected_y].astype(int).value_counts()
                st.subheader("Contagem de Classes na Coluna Selecionada")
                st.table(class_counts)
                
                
                st.subheader("Random Forest Classifier")
                rf_model, rf_predictions = train_random_forest(X_train_scaled, y_train,
                                                               X_test_scaled, n_estimators, max_depth, min_samples_split, min_samples_leaf)
                display_metrics("Random Forest Classifier", y_test,rf_predictions,
                                df[selected_y].unique(), inverse_mapping= inverse_mapping)
                display_majority_percentage(y_test, rf_predictions, inverse_mapping)
                st.subheader("Feature Importance - Random Forest")
                plot_feature_importance(rf_model, np.array(selected_x))
                
                
                st.subheader("Gradient Boosting Classifier")
                gb_model, gb_predictions = train_gradient_boosting(X_train_scaled, y_train,
                                                                   X_test_scaled, n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf)
                display_metrics("Gradient Boosting Classifier",y_test, gb_predictions,
                                df[selected_y].unique(), inverse_mapping= inverse_mapping)
                display_majority_percentage(y_test, gb_predictions, inverse_mapping)
                st.subheader("Feature Importance - Gradient Boosting")
                plot_feature_importance(gb_model, np.array(selected_x))
                
                
                st.subheader("Decision Tree Classifier")
                dt_model, dt_predictions = train_decision_tree(X_train_scaled, y_train,
                                                               X_test_scaled, max_depth, min_samples_split, min_samples_leaf)
                display_metrics("Decision Tree Classifier", y_test,
                                dt_predictions, df[selected_y].unique(), inverse_mapping= inverse_mapping)
                display_majority_percentage(y_test, dt_predictions, inverse_mapping)
                st.subheader("Feature Importance - Decision Tree Classifier")
                plot_feature_importance(dt_model, np.array(selected_x))

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
            else:
                st.error("Falha ao dividir os dados.")
    else:
        # Ajuste de HyperParâmetros
        n_estimators = st.slider("Número de Estimadores (Random Forest e Gradient Boosting)", 10, 500, 200, 10)
        max_depth = st.slider("Profundidade Máxima (Random Forest, Decision Tree Classifier e Gradient Boosting)", 1, 20, 10, 1)
        min_samples_split = st.slider("Número Mínimo de Amostras para Dividir (Random Forest, Decision Tree Classifier e Gradient Boosting)", 2, 20, 2, 1)
        min_samples_leaf = st.slider("Número Mínimo de Amostras em Folha (Random Forest, Decision Tree Classifier e Gradient Boosting)", 1, 20, 1, 1)
        learning_rate = st.slider("Taxa de Aprendizado (Gradient Boosting)", 0.01, 1.0, 0.2, 0.01)
        
        if st.button("Executar Modelos"):
            st.header("Treinamento e Avaliação do Modelo")
            
            # Mapeamento e tratamento dos dados
            df, format_mapping, format_mapping_inverse = map_formats(df)
            df, imprint_mapping, imprint_mapping_inverse = map_imprints(df)
            df = process_artists(df)
            
            # Mapeamentos inversos correspondentes às colunas alvo
            inverse_mapping_dict = {
                'Rating_Age': rating_age_mapping_inverse,
                'Format': format_mapping_inverse,
                'Imprint': imprint_mapping_inverse
            }
            
            # Obtendo o mapeamento inverso correspondente à coluna alvo escolhida
            inverse_mapping = inverse_mapping_dict[selected_y]
            
            # Removendo linhas com valores ausentes
            df.dropna(subset=['final_year', 'Price', 'publish_year', 'publish_month', 'publish_day', 'Rating_Age', 'Imprint', 'Format', 'cover_artist', 'writer', 'penciler'], inplace=True)

            X_train, X_test, y_train, y_test = split_data(df, selected_x, selected_y)

            if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
                X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

                # Contagem de classes na coluna selecionada
                class_counts = df[selected_y].astype(int).value_counts()
                st.subheader("Contagem de Classes na Coluna Selecionada")
                st.table(class_counts)
                
                
                st.subheader("Random Forest Classifier")
                rf_model, rf_predictions = train_random_forest(X_train_scaled, y_train,
                                                               X_test_scaled, n_estimators, max_depth, min_samples_split, min_samples_leaf)
                display_metrics("Random Forest Classifier", y_test,rf_predictions,
                                df[selected_y].unique(), inverse_mapping= inverse_mapping)
                display_majority_percentage(y_test, rf_predictions, inverse_mapping)
                st.subheader("Feature Importance - Random Forest")
                plot_feature_importance(rf_model, np.array(selected_x))
                
                
                st.subheader("Gradient Boosting Classifier")
                gb_model, gb_predictions = train_gradient_boosting(X_train_scaled, y_train,
                                                                   X_test_scaled, n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf)
                display_metrics("Gradient Boosting Classifier",y_test, gb_predictions,
                                df[selected_y].unique(), inverse_mapping= inverse_mapping)
                display_majority_percentage(y_test, gb_predictions, inverse_mapping)
                st.subheader("Feature Importance - Gradient Boosting")
                plot_feature_importance(gb_model, np.array(selected_x))
                
                
                st.subheader("Decision Tree Classifier")
                dt_model, dt_predictions = train_decision_tree(X_train_scaled, y_train,
                                                               X_test_scaled, max_depth, min_samples_split, min_samples_leaf)
                display_metrics("Decision Tree Classifier", y_test,
                                dt_predictions, df[selected_y].unique(), inverse_mapping= inverse_mapping)
                display_majority_percentage(y_test, dt_predictions, inverse_mapping)
                st.subheader("Feature Importance - Decision Tree Classifier")
                plot_feature_importance(dt_model, np.array(selected_x))
                
                
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
            else:
                st.error("Falha ao dividir os dados.")

if __name__ == '__main__':
    main()
    