import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#recursos NLTK necessários
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

#função para prè-processar o texto do tweet

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    #convertendo para minúsculo
    text = text.lower()

    #removendo URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    #removendo menções e hashtags (manter o conteúdo)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    #removendo caracteres especiais
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #removendo espaços extras
    text = re.sub(r'\s+', ' ', text).strip()

    #tokenização
    tokens = word_tokenize(text)

    #removendo stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    #lematização
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)
