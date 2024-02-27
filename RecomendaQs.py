import streamlit as st

st.set_page_config(
    page_title = "PISI3 - BSI - UFRPE por Grupo 4",
    layout = "wide",
    menu_items = {
        'About': '''Este sistema foi desenvolvido pelo grupo 4, para a disciplina de 
        Projeto Interdisciplinar para Sistemas de Informação 3 (PISI3) do 3° período do curso de Bacharelado em Sistemas de Informação
        (BSI) da Universidade Federal Rural de Pernambuco (UFRPE). Visando o desenvolvimento do aplicativo "RecomendaQs", que usa análise
        de dados e Machine Learning para a recomendação do quadrinho.
        '''
    }
)

st.markdown(f'''
    <h1>RecomendaQs</h1>
    <br>
    Este sistema foi desenvolvido pelo grupo 4, para a disciplina de 
    Projeto Interdisciplinar para Sistemas de Informação 3 (PISI3) do 3° período do curso de Bacharelado em Sistemas de Informação
    (BSI) da Universidade Federal Rural de Pernambuco (UFRPE). Visando o desenvolvimento do aplicativo "RecomendaQs", que usa análise
    de dados e Machine Learning para a recomendação do quadrinho.
    <br>     
    <br>
    Participantes do grupo
    <ul>
            <li>Carlos Vinicios</li>
            <li>Leandro dos Santos</li>
            <li>Leon Lourenço</li>
            <li>Vithória Camila</li>
''', unsafe_allow_html=True)
