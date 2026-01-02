import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_text

# Configurações
MAX_WORDS = 10000
MAX_LEN = 50
EMBEDDING_DIM = 100

DATA_FILE = 'train.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'LSTM_Simples_best_model.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pickle')

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print(" Carregando dados...")
    df = pd.read_csv(DATA_FILE)
    df['processed'] = df['text'].apply(preprocess_text)
    df = df[df['processed'] != 'empty_text']
    print(f" {len(df)} tweets")
    return df

#Baseline com TF-IDF + Logistic Regression
def train_baseline_tfidf(df):
    print("\n" + "=" * 70)
    print(" BASELINE: TF-IDF + Logistic Regression")
    print("=" * 70)

    X_train, X_val, y_train, y_val = train_test_split(
        df['processed'], df['target'],
        test_size=0.2, random_state=42, stratify=df['target']
    )

    # TF-IDF com bi-gramas e tri-gramas
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    # Avaliar
    train_acc = model.score(X_train_vec, y_train)
    val_acc = model.score(X_val_vec, y_val)

    print(f"\nTreino Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")

    # Report detalhado
    y_pred = model.predict(X_val_vec)
    print("\n Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Não-Desastre', 'Desastre']))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\n Confusion Matrix:")
    print(f"                Pred: Não-Desastre  Pred: Desastre")
    print(f"Real: Não-Desastre      {cm[0, 0]:5d}            {cm[0, 1]:5d}")
    print(f"Real: Desastre          {cm[1, 0]:5d}            {cm[1, 1]:5d}")

    # Salvar modelos TF-IDF
    with open('models/tfidf_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/lr_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    print("\nModelos TF-IDF salvos em:")
    print("   - models/tfidf_vectorizer.pickle")
    print("   - models/lr_model.pickle")
    return vectorizer, model

    #CNN em vez de LSTM - melhor para n-gramas
def build_cnn_model(vocab_size):
    print("\n" + "=" * 70)
    print("CNN MODEL (melhor para padrões locais)")
    print("=" * 70)

    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Dropout(0.2),

        # Múltiplos filtros CNN para capturar diferentes n-gramas
        Conv1D(128, 3, activation='relu'),  # Tri-gramas
        Conv1D(128, 4, activation='relu'),  # 4-gramas
        Conv1D(128, 5, activation='relu'),  # 5-gramas

        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.build(input_shape=(None, MAX_LEN))
    model.summary()

    return model

def train_cnn_model(df):
    print("\n" + "=" * 70)
    print(" TREINANDO CNN")
    print("=" * 70)

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed'])

    # Salvar tokenizer
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index)) + 1
    print(f"Vocabulário: {vocab_size}")

    # Preparar dados
    sequences = tokenizer.texts_to_sequences(df['processed'])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    y = df['target'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelo
    model = build_cnn_model(vocab_size)

    # Treinar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Salvar
    model.save(MODEL_PATH)
    print(f"\n Modelo salvo")

    # Avaliar
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n Val Accuracy: {acc:.4f} ({acc * 100:.1f}%)")

    # Teste
    test_texts = [
        ("earthquake destroys buildings", True),
        ("fire emergency evacuation", True),
        ("happy birthday party", False),
        ("love beautiful day", False),
        ("burning up the dance floor", False),  # Figurativo!
        ("people killed in explosion", True),
    ]

    print("\nTESTE:")
    correct = 0
    for text, expected in test_texts:
        proc = preprocess_text(text)
        seq = tokenizer.texts_to_sequences([proc])
        pad = pad_sequences(seq, maxlen=MAX_LEN)
        pred = model.predict(pad, verbose=0)[0][0]
        predicted = pred > 0.5

        status = "OK" if predicted == expected else "NÃO OK"
        correct += (predicted == expected)

        print(f"{status} {text:35s} → {pred:.4f}")

    print(f"\nAcurácia: {correct}/{len(test_texts)}")

    return tokenizer, model


def main():
    print("\n" + "=" * 70)
    print(" TREINAMENTO COMPLETO")
    print("=" * 70)

    df = load_data()

    # 1. Baseline TF-IDF
    vectorizer, lr_model = train_baseline_tfidf(df)

    # 2. CNN Model
    tokenizer, cnn_model = train_cnn_model(df)

    print("\n" + "=" * 70)
    print(" CONCLUÍDO!")
    print("=" * 70)
    print("\n O modelo TF-IDF geralmente funciona MELHOR para este tipo de problema")
    print("   porque captura melhor o contexto com n-gramas")

if __name__ == "__main__":
    main()