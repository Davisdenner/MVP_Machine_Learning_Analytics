import pickle
import numpy as np
from preprocess import preprocess_text
from deep_translator import GoogleTranslator

# constantes
MAX_LEN = 100  # mesmo valor usado no treinamento

def load_tfidf_model():
    try:
        with open('models/tfidf_vectorizer.pickle', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/lr_model.pickle', 'rb') as f:
            model = pickle.load(f)
        print(" Modelo TF-IDF carregado com sucesso!")
        return vectorizer, model
    except Exception as e:
        print(f" Erro ao carregar modelo: {e}")
        raise


def predict_tweet(text, vectorizer=None, model=None, translate=True):

    if vectorizer is None or model is None:
        vectorizer, model = load_tfidf_model()

    text_en = text

    # Tradução
    if translate:
        translator = GoogleTranslator(source='pt', target='en')
        try:
            text_en = translator.translate(text)
            print(f"Texto original (PT): {text}")
            print(f"Texto traduzido (EN): {text_en}")
        except Exception as e:
            print(f"Erro na tradução: {e}")
            text_en = text

    # Preprocessamento
    processed_text = preprocess_text(text_en)
    print(f"Texto processado: {processed_text}")

    if not processed_text or processed_text == 'empty_text':
        return {
            'text': text,
            'text_translated': text_en,
            'processed_text': processed_text,
            'is_disaster': False,
            'probability': 0.0,
            'class': 'Erro - Texto vazio'
        }

    # Vetorizar
    X = vectorizer.transform([processed_text])

    # Predição
    probability = model.predict_proba(X)[0][1]  # Probabilidade da classe 1 (desastre)
    is_disaster = bool(probability > 0.5)

    return {
        'text': text,
        'text_translated': text_en,
        'processed_text': processed_text,
        'is_disaster': is_disaster,
        'probability': float(probability),
        'class': 'Desastre' if is_disaster else 'Não Desastre'
    }


def test_predictions():
    """Testa com exemplos variados"""
    vectorizer, model = load_tfidf_model()

    test_cases = [
        "Terremoto devastador atinge a cidade",
        "Que dia lindo para um passeio no parque",
        "Incêndio florestal fora de controle",
        "Adorei o novo filme que assisti ontem",
        "Tsunami alerta emitido para a costa",
        "Explosão em prédio causa várias mortes",
        "Celebrando meu aniversário com amigos",
        "Furacão categoria 5 se aproxima da costa",
        "Assistindo Netflix em casa",
        "Evacuação de emergência ordenada",
    ]

    print("\n" + "=" * 70)
    print(" TESTE DE PREDIÇÕES COM TF-IDF")
    print("=" * 70)

    for text in test_cases:
        result = predict_tweet(text, vectorizer, model)
        emoji = "OK" if result['is_disaster'] else "NOT OK"
        print(f"\n{emoji} Texto: {result['text']}")
        print(f"   Classe: {result['class']}")
        print(f"   Probabilidade: {result['probability']:.4f}")
        print("-" * 70)


if __name__ == "__main__":
    test_predictions()