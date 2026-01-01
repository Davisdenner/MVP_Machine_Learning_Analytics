#(versão atualizada para TensorFlow 2.20)
import streamlit as st
import os
from tensorflow.keras.models import load_model
import pickle
from predict import predict_tweet

st.set_page_config(
    page_title="Classificador de Tweets de Desastre Naturais",
    layout="wide",
)

# ===================== CSS PERSONALIZADO =====================
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
        margin-top: 10px;
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
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }

        div.stButton > button {
            padding: 1rem 1.5rem;
            min-height: 48px;
        }

        .language-warning {
            padding: 0.875rem 1rem;
        }

        .stTextArea textarea {
            min-height: 100px;
        }
    }

    @media (min-width: 769px) {
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 320px;
        }

        .stTextArea textarea {
            min-height: 120px;
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

st.title("Classificador de Tweets de Desastre Naturais")
st.write("Digite um tweet para classificar se está relacionado a um desastre natural real ou não")

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

        with st.expander("Ver detalhes"):
            st.write("Texto processado:", result['processed_text'])
    else:
        st.error("Por favor, digite um tweet para classificar.")
