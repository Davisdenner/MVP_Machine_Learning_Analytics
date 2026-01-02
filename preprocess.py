import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Baixando recursos necessários
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Stopwords que NÃO devem ser removidas em contexto de desastres
KEEP_STOPWORDS = {
    'no', 'not', 'nor', 'out', 'off', 'over', 'under',
    'down', 'up', 'through', 'after', 'before', 'during'
}

def preprocess_text(text, remove_stopwords=True, lemmatize=True):

    if not isinstance(text, str):
        return ""

    # Salvando texto original para debug
    original_text = text

    # Convertendo para minúsculo
    text = text.lower()

    # Removendo URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Removendo menções (@usuario)
    text = re.sub(r'@\w+', '', text)

    # Removendo # mas mantendo o conteúdo da hashtag
    text = re.sub(r'#', '', text)

    # Mantendo alguns números importantes (anos, magnitudes, etc)
    # Remove apenas números isolados, mantém se fizer parte de palavra/contexto
    text = re.sub(r'\b\d+\b', '', text)  # Remove números isolados

    # Removendo pontuação, mas preservando espaços
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Removendo espaços extras
    text = re.sub(r'\s+', ' ', text).strip()

    # Verificação: se texto ficou vazio, retorna early
    if not text or len(text.strip()) == 0:
        return "empty_text"

    # Tokenização
    tokens = text.split()

    # Removendo stopwords de forma mais inteligente
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words('english')) - KEEP_STOPWORDS
            # Mantém palavras curtas se forem importantes ou não forem stopwords
            tokens = [
                word for word in tokens
                if word not in stop_words or len(word) <= 2
            ]
        except Exception as e:
            print(f"Aviso: Erro ao carregar stopwords - {e}")
            # Fallback: mantém palavras com mais de 1 letra
            tokens = [word for word in tokens if len(word) > 1]

    # Lematização
    if lemmatize:
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]  # Verbos
            tokens = [lemmatizer.lemmatize(word, pos='n') for word in tokens]  # Substantivos
        except Exception as e:
            print(f"Aviso: Erro na lematização - {e}")

    # Removendo tokens vazios ou muito curtos
    tokens = [word for word in tokens if word and len(word) > 1]

    # Retorna o texto processado
    processed = ' '.join(tokens)

    # Verificação final: garantir que não está vazio
    if not processed or len(processed.strip()) == 0:
        return "empty_text"

    return processed


def test_preprocessing():
    test_cases = [
        "Devastating earthquake hits the city",
        "Forest fire out of control",
        "Tsunami warning issued for coast",
        "What a beautiful day for a walk in the park",
        "BREAKING: Hurricane category 5 approaching Florida!",
        "Just had lunch with @friend #yummy",
        "Building on fire! Emergency services called!!!",
        "",
        "123 456",
    ]

    print("=" * 70)
    print("TESTE DE PREPROCESSAMENTO")
    print("=" * 70)

    for text in test_cases:
        processed = preprocess_text(text)
        print(f"\nOriginal:    {text}")
        print(f"Processado:  {processed}")
        print("-" * 70)

if __name__ == "__main__":
    test_preprocessing()