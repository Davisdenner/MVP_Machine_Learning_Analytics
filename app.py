#(versão atualizada para TensorFlow 2.20)
import streamlit as st
import os
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

with st.expander("**Entendendo as Técnicas: TF-IDF e N-gramas**", expanded=False):
    st.markdown("""
    ### **O que é TF-IDF?**

    **TF-IDF** significa **Term Frequency-Inverse Document Frequency** (Frequência do Termo - Frequência Inversa do Documento).

    **Em termos simples**: É uma técnica que mede **quão importante** uma palavra é em um texto específico, comparando com todos os outros textos.

    **Como funciona:**
    - **TF (Term Frequency)**: Conta quantas vezes uma palavra aparece no texto
    - **IDF (Inverse Document Frequency)**: Verifica se a palavra é rara ou comum em todos os textos
    - **Resultado**: Palavras **raras mas frequentes** no texto atual ganham **pontuação alta**

    **Exemplo prático:**
    - Palavra "terremoto" em um tweet sobre desastre = **ALTA pontuação** (importante e específica)
    - Palavra "o", "de", "para" = **BAIXA pontuação** (muito comum, pouco informativa)

    ### **O que são N-gramas?**

    **N-gramas** são **sequências de palavras** que o modelo analisa juntas, não apenas palavras isoladas.

    **Tipos usados no projeto:**
    - **1-grama (unigrama)**: palavra isolada → *"incêndio"*
    - **2-grama (bigrama)**: duas palavras juntas → *"incêndio florestal"*  
    - **3-grama (trigrama)**: três palavras juntas → *"incêndio florestal descontrolado"*

    **Por que isso é importante:**
    - **Contexto**: "Burning down" (desastre) vs "burning calories" (exercício)
    - **Precisão**: "Emergency evacuation" é muito mais específico que apenas "emergency"
    - **Semântica**: Captura o **significado real** da frase, não só palavras soltas

    ### **Por que TF-IDF + N-gramas é eficiente?**

     **Rápido**: Processamento muito mais veloz que Deep Learning  
    **Eficaz**: Captura padrões textuais importantes  
    **Interpretável**: Podemos ver exatamente quais palavras/frases influenciam a decisão  
    **Menos recursos**: Não precisa de GPU ou grandes quantidades de memória  
    **Produção**: Ideal para aplicações reais que precisam de respostas rápidas  

    ###  **Exemplo no seu Tweet:**

    **Tweet**: *"Terremoto devastador atinge a cidade"*

    **O modelo analisa:**
    - **1-gramas**: "terremoto" (alta pontuação), "devastador" (alta), "atinge" (média), "cidade" (média)
    - **2-gramas**: "terremoto devastador" (altíssima pontuação - muito específico!)
    - **3-gramas**: "terremoto devastador atinge" (contexto completo de emergência)

    **Resultado**: Classificação **"Desastre"** com alta confiança! 
    """)

with st.expander("O que este modelo faz e quando utilizar", expanded=False):
    st.markdown("""
    **O que o modelo faz**  
    Esta aplicação utiliza um modelo de Machine Learning baseado em 
    **TF-IDF (Term Frequency-Inverse Document Frequency)** e **Regressão Logística** 
    para classificar tweets de acordo com a presença de **eventos reais de desastre natural**, 
    analisando a **importância estatística** das palavras e **n-gramas** no texto.

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

    - **Acurácia**: 85%  
    - **Precisão (Desastre)**: 82%  
    - **Recall (Desastre)**: 87%  
    - **F1-score**: 84%  

    **Interpretação das métricas**  
    - **Recall** indica a capacidade do modelo de identificar corretamente
      tweets que representam **eventos reais de desastre**.
    - **F1-score** representa o equilíbrio entre **Precisão** e **Recall**,
      sendo especialmente relevante em cenários com classes desbalanceadas.

    > O modelo TF-IDF apresenta excelente performance para este tipo de classificação
    textual, aproveitando padrões de frequência de palavras e n-gramas.
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
    st.header("Sobre o Projeto")

    st.markdown("""
    **Contexto Acadêmico:**
    Desenvolvido para o MVP em Machine Learning & Analytics da PUC-Rio.

    **Objetivo:**
    Classificar tweets relacionados a desastres naturais usando NLP.

    **Diferencial:**
    Comparação empírica entre Deep Learning e métodos tradicionais de ML, demonstrando que TF-IDF pode superar redes neurais em problemas específicos.
    """)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.header("Informações do Modelo")

    # Status do modelo atual
    st.success("**Modelo Ativo:** TF-IDF + Regressão Logística")

    st.header("Performance vs. Alternativas")

    # Comparação visual
    comparison_data = {
        "TF-IDF + LogReg": {"acc": 85, "speed": "< 1s", "resources": "Baixo"},
        "LSTM Bidirecional": {"acc": 82, "speed": "~5s", "resources": "Alto"},
        "LSTM Simples": {"acc": 80, "speed": "~3s", "resources": "Médio"}
    }

    for model, metrics in comparison_data.items():
        if model == "TF-IDF + LogReg":
            st.success(f"**{model}**")
            st.write(
                f" Acurácia: {metrics['acc']}% |  Velocidade: {metrics['speed']} | Recursos: {metrics['resources']}")
        else:
            st.info(f"**{model}**")
            st.write(
                f"Acurácia: {metrics['acc']}% |  Velocidade: {metrics['speed']} | Recursos: {metrics['resources']}")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.header("Tecnologias Utilizadas")

    # Badges das tecnologias principais
    tech_badges = [
        ("Python", "🐍"),
        ("scikit-learn", "📊"),
        ("NLTK", "📝"),
        ("Streamlit", "⚡"),
        ("TF-IDF", "🔤"),
        ("Deep Translator", "🌐")
    ]

    for tech, emoji in tech_badges:
        st.markdown(f"{emoji} **{tech}**")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # Footer com informações do desenvolvedor
    st.markdown(
        """
        <div class='sidebar-footer'>
        <h4>Desenvolvedor</h4>
        <strong>Davis Denner Costa Silva</strong><br>
        Machine Learning & Analytics - PUC-Rio<br><br>
        <a href="https://www.linkedin.com/in/davis-denner-costa-silva-4536a51b0" target="_blank">
        LinkedIn</a> | 
        <a href="https://github.com/Davisdenner" target="_blank">
        GitHub</a>
        <br><br>
        <small> <em>Demonstrando que nem sempre "mais complexo" significa "melhor".</em></small>
        </div>
        """,
        unsafe_allow_html=True
    )
# ===================== FUNÇÕES PARA CARREGAR MODELO E TOKENIZER =====================#
@st.cache_resource
def carregar_modelos_tfidf():
    try:
        with open('models/tfidf_vectorizer.pickle', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/lr_model.pickle', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        st.error(f"Erro ao carregar modelos TF-IDF: {e}")
        return None, None

# Carregando modelos TF-IDF ao iniciar a aplicação
vectorizer, model = carregar_modelos_tfidf()

tweet_input = st.text_area("Texto do Tweet:", height=100)

if st.button("Classificar Tweet"):
    if tweet_input:
        if vectorizer is not None and model is not None:
            with st.spinner("Classificando com modelo TF-IDF..."):
                result = predict_tweet(tweet_input, vectorizer, model)

            st.subheader("Resultado:")

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Classificação: {result['class']}")
            with col2:
                st.info(f"Probabilidade: {result['probability']:.2%}")
                st.progress(result['probability'])
                st.caption('Probabilidades inferiores a 50% são automaticamente classificadas como "Não Desastre".')
            with st.expander("Ver detalhes"):
                st.write("**Texto original:**")
                st.code(result['text'])
                if result['text_translated'] != result['text']:
                    st.write("**Texto traduzido:**")
                    st.code(result['text_translated'])
                st.write("**Texto após pré-processamento:**")
                st.code(result['processed_text'])
        else:
            st.error("Modelos TF-IDF não puderam ser carregados. Verifique se os arquivos estão na pasta models/")
    else:
        st.error("Por favor, digite um tweet para classificar.")
