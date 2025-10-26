#(versão atualizada para TensorFlow 2.20)
import streamlit as st
import os
from tensorflow.keras.models import load_model
import pickle
from predict import predict_tweet



st.set_page_config(
    page_title="Classificador de Tweets de Desastre Naturais",
)


st.title("Classificador de Tweets de Desastre Naturais")
st.write("Digite um tweet para classificar se está relacionado a um desastre real ou não")


modelo_selecionado = st.sidebar.selectbox(
    "Selecione o modelo:",
    ["LSTM_Simples", "LSTM_Bidirecional"]
)


#carregando modelo selecionado
@st.cache_resource
def carregar_modelo(nome_modelo):
    caminho_modelo = f"models/{nome_modelo}_best_model.keras"
    if os.path.exists(caminho_modelo):
        return load_model(caminho_modelo)
    else:
        st.error(f"Modelo {caminho_modelo} não encontrado!")
        return None


#carregando tokenizer
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

#campo para entrada do tweet
tweet_input = st.text_area("Texto do Tweet:", height=100)

if st.button("Classificar Tweet"):
    if tweet_input:
        with st.spinner(f"Classificando com modelo {modelo_selecionado}..."):
            #passando o tokenizer para a função predict_tweet
            from predict import predict_tweet
            result = predict_tweet(tweet_input, tokenizer)

        st.subheader("Resultado:")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"Classificação: {result['class']}")

        with col2:
            #formatando a probabilidade como porcentagem
            st.info(f"Probabilidade: {result['probability']:.2%}")

            #mostrando barra de progresso para visualização
            st.progress(result['probability'])

        #mostrando detalhes adicionais
        with st.expander("Ver detalhes"):
            st.write("Texto processado:", result['processed_text'])
    else:
        st.error("Por favor, digite um tweet para classificar.")

#informações no sidebar
st.sidebar.header("Sobre o Projeto")
st.sidebar.write(
    "Este app utiliza um modelo de Deep Learning para classificar "
    "tweets relacionados a desastres naturais."
)
st.sidebar.write("Desenvolvido como parte do MVP em Machine Learning & Analytics.")