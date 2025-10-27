
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Garantindo que os recursos NLTK necessários estejam disponíveis
# Baixando recursos de forma mais explícita, mesmo que já estejam presentes
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# A versão anterior que apenas verifica não é suficiente para o Streamlit Cloud
# for resource in ["stopwords", "wordnet", "punkt"]:
#     try:
#         if resource == "punkt":
#             nltk.data.find(f"tokenizers/{resource}")
#         else:
#             nltk.data.find(f"corpora/{resource}")
#     except LookupError:
#         nltk.download(resource, quiet=True)

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

    # Tokenização
    tokens = word_tokenize(text)

    # Removendo stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Lematização
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)