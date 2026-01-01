import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Baixando recursos de forma explícita
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# Função para pré-processar o texto do tweet
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Convertendo para minúsculo
    text = text.lower()

    # Removendo URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Removendo menções (mas mantendo o resto)
    text = re.sub(r'@\w+', '', text)

    # Removendo # mas mantendo o conteúdo da hashtag
    text = re.sub(r'#', '', text)

    # Removendo números e caracteres especiais (mas mantendo letras e espaços)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Removendo espaços extras
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenização simples
    tokens = text.split()

    # Removendo stopwords (mas de forma menos agressiva)
    try:
        stop_words = set(stopwords.words('english'))
        # IMPORTANTE: Mantém palavras com menos de 3 letras se forem importantes
        tokens = [word for word in tokens if word not in stop_words or len(word) > 2]
    except:
        # Fallback se as stopwords não estiverem disponíveis
        tokens = [word for word in tokens if len(word) > 1]

    # Lematização
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except:
        # Fallback se a lematização falhar
        pass

    # Retorna o texto processado
    processed = ' '.join(tokens)

    # IMPORTANTE: Se o texto ficar vazio, retorna uma string com pelo menos uma palavra
    if not processed or len(processed.strip()) == 0:
        return "empty"

    return processed