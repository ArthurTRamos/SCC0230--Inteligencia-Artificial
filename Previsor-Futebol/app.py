import streamlit as st
import pandas as pd
import numpy as np
import base64

# Configurações da Página
st.set_page_config(page_title="Vidente do Futebol", # Titulo da Guia
                   layout="wide", # Modelo de renderização
                   page_icon="⚽", # Icone da Guia
                   initial_sidebar_state="expanded", # Estado default da barra literal = expandida (aberta)
                   )

# CSS Injection
## Remover os Links
st.markdown("""
    <style>
    /* Mira exatamente no container do link (corrente) usando o ID de teste do Streamlit */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
## Remover as marcas do streamlit
# st.markdown("""
#     <style>
#     /* Esconde o menu sanduíche (três pontinhos) no topo */
#     #MainMenu {visibility: hidden;}
#     
#     /* Esconde o botão 'Deploy' se aparecer */
#     [data-testid="stAppDeployButton"] {
#         display: none !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

# Definição de Funções
## Função de codificação base64 para as imagens
@st.cache_data # Uso de cache para otimização
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

## Modelos
def M1(a, b):
    return a

def M2(a, b):
    return b

def M3(a,b):
    array = [a,b]
    return sorted(array)[0]

def M4(a, b):
    return np.random.choice([a,b])
    
# Dados
## Dataframe Modelos
df_model = pd.DataFrame({
    'Modelo': ['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4'],
    'Function': [M1, M2, M3, M4],
    'Desc': ["O modelo 1 sempre retorna o primeiro.",
             "O modelo 2 sempre retorna o segundo.",
             "O modelo 3 retorna o primeiro em ordem alfabética.",
             "O modelo 4 retorna um aleatório."]
}).set_index('Modelo')

## Dataframe Times
df_times = pd.DataFrame({'Time': ['Atlético Mineiro', 'Bahia', 'Botafogo', 'Ceará', 'Corinthians', 'Cruzeiro', 'Flamengo', 'Fluminense',
                                  'Fortaleza', 'Grêmio', 'Internacional', 'Juventude', 'Mirassol', 'Palmeiras', 'Bragantino', 'Santos',
                                  'São Paulo', 'Sport', 'Vasco da Gama', 'Vitória']})

# Site
## Logo
st.logo("Imgs/icmc-dark.png")

## Sidebar
sidebar = st.sidebar # Cria a sidebar
with sidebar:
    st.header(f"Escolha um de nossos {df_model.shape[0]} modelos para a previsão:") # Adiciona o header a sidebar com o n° de modelos disponíveis
    selected_model = st.selectbox('Modelo: ', df_model.index) # Caixa de seleção de modelo
    st.markdown(df_model.loc[selected_model, 'Desc']) # Descrição do modelo escolhido

## Seleciona a função baseada no modelo escolhido
selected_function = df_model.loc[selected_model, 'Function']

# Seletor de Times
col1, col_vs, col2 = st.columns([4, 1, 4]) # Define as colunas

## Coluna Esquerda (Time da Casa)
with col1:
    # Título
    st.markdown("<h3 style='text-align: center;'>Casa</h3>", unsafe_allow_html=True)

    # Select do time
    TimeA = st.selectbox('Time da Casa:', df_times['Time'], label_visibility="collapsed")

    # Brasão do Time
    st.markdown(
        f"<div style='text-align: center;'><img src='data:image/png;base64,{img_to_base64(f"crests/{TimeA}.png")}' height='300'></div>",
        unsafe_allow_html=True
    )
    
## Coluna central (VS)
with col_vs:
    st.markdown("<h1 style='text-align: center; padding-top: 230px;'>VS</h1>", unsafe_allow_html=True)

## Coluna Direita (Time Visitante)
with col2:
    # Título
    st.markdown("<h3 style='text-align: center;'>Visitante</h3>", unsafe_allow_html=True)

    # Select do time
    TimeB = st.selectbox('Time Visitante:', df_times.loc[df_times['Time'] != TimeA], label_visibility="collapsed")

    # Brasão do time
    st.markdown(
        f"<div style='text-align: center;'><img src='data:image/png;base64,{img_to_base64(f"crests/{TimeB}.png")}' height='300'></div>",
        unsafe_allow_html=True
    )

# Botão  
st.markdown("---") # Divisória
predicted_bool = st.button("Prever Resultado") # Botão para prever o resultado
# detail_bool = st.checkbox("Mostrar Detalhes")

# Diplay condicional do resultado
if(predicted_bool):
    resultado = selected_function(TimeA, TimeB) # Calcula o resultado aplicando a função escolhida
    with st.container(border=True): # Container para o "cartão" de apresentação do resultado
        res_logo, res_text = st.columns([1, 3]) # Criação das colunas

        # Na 1° coluna se adiciona a logo do time
        with res_logo: 
            # Logo do time vencedor
            st.markdown(
                f"<div style='text-align: center; padding-bottom: 15px;'><img src='data:image/png;base64,{img_to_base64(f"crests/{resultado}.png")}' height='150'></div>",
                unsafe_allow_html=True
            )

        # Na 2° coluna se coloca o anúncio da vitória (texto)
        with res_text:
            st.title(f"Vencedor: {resultado}") # Texto resultado
            st.caption(f"Previsão de acordo com {selected_model}") # Legenda mostra o modelo responsável pela previsão
    



