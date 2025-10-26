
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess_text

# constantes
MAX_LEN = 100  # mesmo valor usado no treinamento

#carregando modelo
try:
    model = load_model('models/LSTM_Simples_best_model.keras')
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    raise

def load_tokenizer():
    try:
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer carregado com sucesso!")
        return tokenizer
    except Exception as e:
        print(f"Erro ao carregar tokenizer: {e}")
        raise

def predict_tweet(text, tokenizer=None):
    #se o tokenizer não for fornecido, tenta carregá-lo
    if tokenizer is None:
        tokenizer = load_tokenizer()
    #pré-processando o texto
    processed_text = preprocess_text(text)
    #convertendo para sequência
    sequence = tokenizer.texts_to_sequences([processed_text])
    #aplicando padding
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    #fazendo predição
    prediction = model.predict(padded, verbose=0)[0][0]
    #determinando a classe
    is_disaster = bool(prediction > 0.5)

    return {
        'text': text,
        'processed_text': processed_text,
        'is_disaster': is_disaster,
        'probability': float(prediction),
        'class': 'Desastre' if is_disaster else 'Não Desastre'
    }