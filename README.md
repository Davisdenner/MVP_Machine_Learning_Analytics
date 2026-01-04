
<p align="center">
  <img src="https://reari.uff.br/wp-content/uploads/sites/171/2023/09/pucrio.png" width="100" height="100"/>
</p>

<h1 align="center">Classificação de Tweets de Desastre com Machine Learning</h1>

Este projeto consiste em um sistema completo de classificação de tweets relacionados a desastres naturais utilizando técnicas de **Machine Learning** e **Processamento de Linguagem Natural (NLP)**. O conjunto de dados utilizado, [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started), foi obtido no **Kaggle** e contém milhares de tweets rotulados.

O projeto implementa uma abordagem híbrida, comparando **Deep Learning** (LSTM) com **métodos tradicionais de ML** (TF-IDF + Regressão Logística), demonstrando que para este tipo de problema, a **vectorização TF-IDF** com **n-gramas** oferece excelente performance com menor complexidade computacional.

## Sobre o Projeto
Este projeto foi desenvolvido como parte do **MVP em Machine Learning & Analytics** da **PUC-Rio**, demonstrando a aplicação prática de técnicas de NLP para problemas reais de classificação de texto.

**Desenvolvido por**: Davis Denner Costa Silva

**LinkedIn**: [davis-denner-costa-silva](https://www.linkedin.com/in/davis-denner-costa-silva-4536a51b0)

**GitHub**: [Davisdenner](https://github.com/Davisdenner)

**Deploy**: [Interface Streamlit](https://classificador-davis-denner-01.streamlit.app/)

## Abordagem Principal: TF-IDF + Regressão Logística

O modelo final utiliza:
- **TF-IDF Vectorization** com n-gramas (1-3) para capturar padrões textuais
- **Regressão Logística** com balanceamento de classes para classificação
- **Pré-processamento inteligente** preservando palavras importantes para contexto de desastres
- **Tradução automática** (PT para EN) para suporte multilíngue

## Principais Características
- **Dataset**: 7.613 tweets de treino e 3.263 de teste
- **Problema**: Classificação binária (desastre vs não-desastre)
- **Modelos comparados**: TF-IDF + LogReg, LSTM simples e LSTM bidirecional
- **Técnicas aplicadas**: Vectorização TF-IDF, pré-processamento de texto, balanceamento de classes
- **Interface Web**: Aplicação Streamlit para demonstração interativa

## Funcionalidades

- **Classificação automática** de tweets em português e inglês
- **Interface web interativa** com visualização de probabilidades
- **Tradução automática** para suporte multilíngue
- **Análise detalhada** mostrando texto processado
- **Comparação de modelos** (TF-IDF vs Deep Learning)

## Metodologia

1. **Pré-processamento**: Limpeza de texto, remoção de ruído, tokenização
2. **Vectorização**: TF-IDF com n-gramas para capturar contexto
3. **Modelagem**: Regressão Logística com balanceamento de classes
4. **Avaliação**: Métricas de classificação e análise de erros
5. **Deploy**: [Interface Streamlit](https://classificador-davis-denner-01.streamlit.app/) para demonstração

## Insights do Projeto

- **TF-IDF superou Deep Learning** para este problema específico
- **N-gramas** são fundamentais para capturar contexto em textos curtos
- **Tradução automática** preserva o significado para classificação
- **Pré-processamento inteligente** melhora significativamente a performance

<div align="center">

## Tecnologias Utilizadas

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" width="40" height="40" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original-wordmark.svg" width="40" height="40"  />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-plain-wordmark.svg" width="50" height="50" /> 
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original.svg" width="40" height="40"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/plotly/plotly-original.svg" width="30" height="30" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" width="40" height="40" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/tensorflow/tensorflow-original.svg" width="30" height="30"  />
<img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" width="30" height="30" />

**Bibliotecas Principais**: scikit-learn, NLTK, deep-translator, streamlit, tensorflow, pandas, numpy

</div>