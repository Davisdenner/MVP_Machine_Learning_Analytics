# Classificação de Tweets de Desastre com Deep Learning
Este projeto consiste em um notebook desenvolvido que tem como objetivo classificar tweets relacionados a desastres naturais utilizando técnicas de Deep Learning e Processamento de Linguagem Natural (NLP). O conjunto de dados utilizado neste projeto contém milhares de tweets rotulados, permitindo treinar modelos neurais para identificar automaticamente conteúdo relacionado a emergências e desastres.
Todas as etapas de pré-processamento de texto, análise exploratória, treinamento de múltiplas arquiteturas de redes neurais e avaliação dos resultados foram conduzidas utilizando bibliotecas como TensorFlow/Keras, NLTK e scikit-learn.

O projeto demonstra a aplicação prática de redes neurais para classificação de texto em contextos de emergência, com potencial uso em sistemas de monitoramento de redes sociais para detecção precoce de desastres.

## Principais Características
- **Dataset**: 7.613 tweets de treino e 3.263 de teste
- **Problema**: Classificação binária (desastre vs não-desastre)
- **Arquiteturas testadas**: LSTM simples, LSTM bidirecional, CNN e modelo híbrido
- **Melhor resultado**: 82.4% de acurácia com LSTM Bidirecional otimizado
- **Técnicas aplicadas**: Tokenização, embedding de palavras, regularização e otimização de hiperparâmetros

## Tecnologias Utilizadas
- **Deep Learning**: TensorFlow, Keras
- **NLP**: NLTK, preprocessamento de texto
- **Análise**: pandas, numpy, scikit-learn
- **Visualização**: matplotlib, seaborn, plotly
