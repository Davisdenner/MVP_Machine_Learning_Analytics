import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Baixando recursos de forma explícita
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Implementação alternativa para tokenização sem depender do punkt_tab
def simple_tokenize(text):
    # Remove pontuação e substitui por espaços
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    # Divide por espaços e filtra tokens vazios
    return [token.strip() for token in text.split() if token.strip()]

# Função para pré-processar o texto do tweet
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Convertendo para minúsculo
    text = text.lower()

    # Removendo URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Removendo menções e hashtags (mantendo o conteúdo)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # Removendo caracteres especiais
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Removendo espaços extras
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenização usando nosso tokenizador simples
    tokens = simple_tokenize(text)

    # Removendo stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    except:
        # Fallback se as stopwords não estiverem disponíveis
        tokens = [word for word in tokens if len(word) > 2]

    # Lematização
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except:
        # Fallback se a lematização falhar
        pass

    return ' '.join(tokens)