from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import streamlit as st
from utils import df_names, read_df

#Data:
#https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
#http://dados.recife.pe.gov.br/dataset/acidentes-de-transito-com-e-sem-vitimas
#

def profile():
    df_name = st.session_state.dataset
    if df_name in st.session_state:
        return 
    df = read_df(df_name)
    profile = ProfileReport(df, title=f"{df_name} Dataset")
    profile.to_file(f"reports/{df_name}.html")
    st.session_state[df_name] = df

def build_header():
    text ='<h1>Análise Exploratória</h1>'+\
    '''<p>Esta página executa o <b>YData Profiling</b> (antigo <i>Pandas Profiling</i>), gerando o relatório html desta ferramenta. 
    Uma vez gerado o HTML, ele é incluído nesta página utilizando <code>streamlit.components.v1</code>.</p>
    <p>Esta lib gera vários gráficos e informações úteis para a análise exploratória de dados dos datasets.</p>
    '''
    st.write(text, unsafe_allow_html=True)

def build_body():
    col1, col2 = st.columns([.3,.7])
    col1.selectbox('Selecione o Dataset', df_names(), label_visibility='collapsed', key='dataset')
    button_placeholder = col2.empty()
    if button_placeholder.button('Analisar'):
        #O container 'col2.empty()' é utilizado para que se substitua o seu conteúdo.
        #Se usar o container diretamente, os conteúdos são adicionados ao invés de serem substituídos.
        button_placeholder.button('Analisando...', disabled=True)
        profile()
        st.experimental_rerun()

def print_report():
    df_name = st.session_state.dataset
    if df_name not in st.session_state:
        return
    st.write(f'Dataset: <i>{df_name}</i>', unsafe_allow_html=True)
    report_file = open(f'reports/{df_name}.html', 'r', encoding='utf-8')
    source_code = report_file.read() 
    components.html(source_code, height=400, scrolling=True)

build_header()
build_body()
print_report()
