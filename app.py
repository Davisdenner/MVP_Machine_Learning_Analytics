#(versão atualizada para TensorFlow 2.20)
import streamlit as st
import os
from tensorflow.keras.models import load_model
import pickle
from predict import predict_tweet

st.set_page_config(
    page_title="Classificador de Tweets de Desastre Naturais",
    layout="wide",
    # Sidebar colapsada por padrão para priorizar conteúdo em telas menores (mobile-first)
    initial_sidebar_state="collapsed"
)
# ===================== CSS PERSONALIZADO =====================#
st.markdown(
    """
    <style>   
    /* Botão */
    div.stButton > button {
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #105a8b;
        cursor: pointer;
    }

    /* ===== SIDEBAR STYLE CORRIGIDO ===== */
    [data-testid="stSidebar"] {
        background: #f6f8fa;
        padding: 25px 20px;
        border-right: 1px solid #dce3ea;
        min-width: 305px;
        max-width: 305px;
    }

    /* Títulos */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-weight: 700;
        color: #2c3e50; 
        margin-top: -20px;
        margin-bottom: 12px;
    }

    /* Parágrafos */
    [data-testid="stSidebar"] p {
        font-size: 0.92rem;
        color: #444;
        line-height: 1.45;
    }

    /* Divisor */
    .sidebar-divider {
        height: 1px;
        background: #d0d7de;
        margin: 22px 0;
    }

    /* ===== MEDIA QUERIES ===== */
    @media (max-width: 769px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
    }


    /* ===== MOBILE (celulares pequenos e médios) ===== */
    @media (max-width: 480px) {
        .main .block-container {
            padding: 0.75rem 0.5rem;
        }
    
        div.stButton > button {
            padding: 0.9rem 1.2rem;
            min-height: 48px; /* acessibilidade */
            font-size: 1rem;
        }
    
        .stTextArea textarea {
            min-height: 90px;
            font-size: 0.95rem;
        }
    }
    
    /* ===== MOBILE GRANDE / TABLET ===== */
    @media (min-width: 481px) and (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.75rem;
        }
    
        div.stButton > button {
            padding: 1rem 1.4rem;
            min-height: 48px;
        }
    
        .stTextArea textarea {
            min-height: 110px;
        }
    }
    
    /* ===== DESKTOP ===== */
    @media (min-width: 769px) {
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 320px;
        }
    
        .stTextArea textarea {
            min-height: 100px;
        }
    }

    /* Footer alinhado e mais bonito */
    .sidebar-footer {
        text-align: center;
        font-size: 0.88rem;
        color: #3a3a3a;
        margin-top: 30px;
    }

    .sidebar-footer a {
        color: #2467c0;
        text-decoration: none;
        font-weight: 600;
    }

    .sidebar-footer a:hover {
        text-decoration: underline;
    }
    """,
    unsafe_allow_html=True
)

# st.title("Classificador de Tweets de Desastre Naturais")
st.markdown(
    """
    <h1 style="margin-top:-50px;">
        Classificador de Tweets de Desastre Naturais
    </h1>
    """,
    unsafe_allow_html=True
)
st.write(
    "Digite um tweet para classificar se o texto descreve um **evento real de emergência** "
    "ou se utiliza linguagem relacionada a desastres fora de um contexto real."
)

st.caption("Dica: use o menu lateral para alterar configurações do modelo e ver detalhes do projeto.")

with st.expander("O que este modelo faz e quando utilizar", expanded=False):
    st.markdown("""
    **O que o modelo faz**  
    Esta aplicação utiliza um modelo de Deep Learning baseado em redes neurais
    recorrentes (**LSTM**) para classificar tweets de acordo com a presença de
    **eventos reais de desastre natural**, analisando o **contexto semântico**
    do texto e não apenas palavras-chave isoladas.
    
    **Suporte a múltiplos idiomas**  
    O modelo foi **treinado originalmente em inglês**. Para permitir o uso da
    aplicação por usuários que escrevem em **português**, foi implementada uma
    etapa automática de **detecção e tradução do texto para o inglês** antes
    da inferência. Esse pré-processamento garante compatibilidade com o modelo treinado,
    mantendo a coerência semântica da mensagem original.
    
    **Contexto de uso**  
    O modelo pode ser utilizado para:
    - Monitoramento de redes sociais em tempo real
    - Apoio à detecção precoce de desastres naturais
    - Análise de grandes volumes de dados textuais para fins de alerta e triagem

    Este projeto tem foco **educacional e demonstrativo**, apresentando uma
    solução prática de NLP.
    """)


with st.expander("O que significam as classificações?",expanded=False):
    st.markdown("""
    **Desastre**  
    Tweets que descrevem a ocorrência real de eventos críticos,
    como enchentes, terremotos, incêndios, deslizamentos ou outras
    situações de emergência.

    **Não Desastre**  
    Tweets que utilizam termos associados a desastres de forma figurativa,
    humorística ou fora de um contexto real de emergência.

    O modelo foi treinado para identificar o **contexto da mensagem**,
    e não apenas palavras-chave isoladas.
    """)

with st.expander("Desempenho do Modelo", expanded=False):
    st.markdown("""
    As métricas abaixo foram obtidas a partir de um **conjunto de validação separado**.

    - **Acurácia**: 82%  
    - **Precisão (Desastre)**: 79%  
    - **Recall (Desastre)**: 85%  
    - **F1-score**: 82%  

    **Interpretação das métricas**  
    - **Recall** indica a capacidade do modelo de identificar corretamente
      tweets que representam **eventos reais de desastre**.
    - **F1-score** representa o equilíbrio entre **Precisão** e **Recall**,
      sendo especialmente relevante em cenários com classes desbalanceadas.

    > O recall da classe **Desastre** foi priorizado para reduzir o risco de
    falsos negativos em situações críticas.
    """)

with st.expander("Limitações do Modelo", expanded=False):
    st.markdown("""
    Embora o modelo apresente bom desempenho geral, algumas limitações devem ser consideradas:

    - Tweets curtos, ambíguos ou com pouco contexto podem gerar classificações menos confiáveis.
    - O modelo foi treinado com dados históricos e pode não generalizar bem para
      gírias, abreviações ou eventos recentes.
    - Ironia, sarcasmo e linguagem figurativa ainda representam desafios para o modelo.
    - A probabilidade exibida representa **confiança estatística**, não uma certeza absoluta.

    Este projeto tem caráter **educacional e demonstrativo**, mas reflete
    desafios reais encontrados em aplicações de NLP em produção.
    """)


with st.sidebar:
    st.header("Configurações do Modelo")
    modelo_selecionado = st.sidebar.selectbox(
        "Selecione o modelo:",
        ["LSTM_Simples", "LSTM_Bidirecional"]
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.header("Sobre o Projeto")
    st.write(
        "Esta aplicação utiliza um modelo de Deep Learning para classificar "
        "tweets relacionados a desastres naturais."
    )
    #st.sidebar.write("Desenvolvido como parte do MVP em Machine Learning & Analytics (PUC-RIO).")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class='sidebar-footer'>
        Desenvolvido por: <strong>Davis Denner</strong><br>
        <a href="https://www.linkedin.com/in/davis-denner-costa-silva-4536a51b0" target="_blank">LinkedIn</a><br>
        <a href="https://github.com/Davisdenner" target="_blank">GitHub</a>
        </p>
        """,
        unsafe_allow_html=True
    )
# ===================== FUNÇÕES PARA CARREGAR MODELO E TOKENIZER =====================#
@st.cache_resource
def carregar_modelo(nome_modelo):
    caminho_modelo = f"models/{nome_modelo}_best_model.keras"
    if os.path.exists(caminho_modelo):
        return load_model(caminho_modelo)
    else:
        st.error(f"Modelo {caminho_modelo} não encontrado!")
        return None

@st.cache_resource
def carregar_tokenizer():
    caminho_tokenizer = "models/tokenizer.pickle"
    if os.path.exists(caminho_tokenizer):
        with open(caminho_tokenizer, 'rb') as handle:
            return pickle.load(handle)
    else:
        st.error("Arquivo tokenizer.pickle não encontrado! Verifique se ele está na pasta models/")
        return None

#carregando tokenizer ao iniciar a aplicação
tokenizer = carregar_tokenizer()


tweet_input = st.text_area("Texto do Tweet:", height=100)

if st.button("Classificar Tweet"):
    if tweet_input:
        with st.spinner(f"Classificando com modelo {modelo_selecionado}..."):
            result = predict_tweet(tweet_input, tokenizer)

        st.subheader("Resultado:")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Classificação: {result['class']}")
        with col2:
            st.info(f"Probabilidade: {result['probability']:.2%}")
            st.progress(result['probability'])
            st.caption('Probabilidades inferiores a 50% são automaticamente classificadas como "Não Desastre".')
        with st.expander("Ver detalhes"):
            st.write("Texto após pré-processamento (remoção de ruído, normalização):")
            st.code(result['processed_text'])
    else:
        st.error("Por favor, digite um tweet para classificar.")
