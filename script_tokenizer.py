
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from preprocess import preprocess_text
import os

# Garantindo que o diretório models existe
if not os.path.exists('models'):
    os.makedirs('models')
    print("Diretório 'models' criado.")

print("Carregando os dados para criar o tokenizer...")

# Carregando os dados de treinamento
train_df = pd.read_csv("https://raw.githubusercontent.com/Davisdenner/MVP---Machine-Learning-Analytics/main/train.csv")

# Pré-processando os textos
print("Pré-processando os textos...")
train_df['processed_text'] = train_df['text'].apply(preprocess_text)

# Removendo textos vazios
train_df = train_df[train_df['processed_text'].str.len() > 0]
print(f"Dados após limpeza: {len(train_df)} tweets")

# Configurações para tokenização - as mesmas do notebook
MAX_FEATURES = 20000  # Tamanho do vocabulário

# Criando e treinando o tokenizer
print("Criando e treinando o tokenizer...")
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['processed_text'])

# Salvando o tokenizer
print("Salvando o tokenizer...")
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Tokenizer salvo com sucesso em 'models/tokenizer.pickle'!")
print(f"Tamanho do vocabulário: {len(tokenizer.word_index)} palavras únicas.")