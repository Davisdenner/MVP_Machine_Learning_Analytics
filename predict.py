import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess_text


#constantes
MAX_LEN = 100 #mesmo valor usado no treinamento


#carregando modelo e tokenizer
model = load_model('model/modelo.h5')

#carregando tokenizer (você precisa salvar isso do seu notebook)
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_tweet(text):

    #pré-processando o texto
    processed_text = preprocess_text(text)

    #convertendo para sequência
    sequence = tokenizer.texts_to_sequences([processed_text])

    #aplicando padding
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    #fazendo predição
    prediction = model.predict(padded)[0][0]

    #determinando a classe
    is_disaster = bool(prediction > 0.5)

    #retornando resultado
    return {
        'text': text,
        'processed_text': processed_text,
        'is_disaster': is_disaster,
        'probability': float(prediction),
        'class': 'Desastre' if is_disaster else 'Não Desastre'
    }
