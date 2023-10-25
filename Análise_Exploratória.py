import streamlit as st

st.set_page_config(
    page_title = "Análise exploratória do dataset escolhido",
    layout = "wide",
    menu_items = {
        'About': '''Este sistema foi desenvolvido pelo prof Gabriel Alves para fins didáticos, para a disciplina de 
        Projeto Interdisciplinar para Sistemas de Informação 3 (PISI3) do 3° período do curso de Bacharelado em Sistemas de Informação
        (BSI) da Universidade Federal Rural de Pernambuco (UFRPE).
        Dúvidas? gabriel.alves@ufrpe.br
        Acesse: bsi.ufrpe.br
        '''
    }
)

st.markdown(f'''
    <h1>Análise exploratória do dataset escolhido</h1>
    <br>
    	O objetivo deste trabalho é criar um sistema que recomenda histórias em quadrinhos com base nas informações 
        obtidas na base de dados “Comic Books  - current values and other data” e preferências do usuário, proporcionando uma experiência 
        personalizada e envolvente. Assim, fazendo recomendações precisas e alinhadas com os gostos individuais de cada usuário.
    <br>
    Alunos envolvidos nesse projeto:
    <ul>
            <li>Carlos Vinicios</li>
            <li>Denys Makene</li>
            <li>Leandro Cesar</li>
            <li>Leon Lourenço</li>
            <li>Vithória Bastos</li>
    </ul>
    Link para o artigo relacionado: <a href="https://docs.google.com/document/d/1Mt5v9iHfDYY7pqOotzms1sehEl7_miA7JX6ik3iAtzw/edit?usp=sharing">
            https://docs.google.com/document/d/1Mt5v9iHfDYY7pqOotzms1sehEl7_miA7JX6ik3iAtzw/edit?usp=sharing</a><br>
''', unsafe_allow_html=True)
