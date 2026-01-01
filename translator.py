from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import streamlit as st

@st.cache_resource

# Criar uma isntância do tradutor - Cached
def get_translator():
    return GoogleTranslator(source='auto', target='en')

def translate_to_english(text, translator=None):
    if translator is None:
        translator = get_translator()

    try:
        # Detectar o idioma e traduzir para inglês se necessário
        detection_lang = detect(text)

        if detection_lang == 'en':
            # Já está em inglês
            return text, False
        else:
            # Traduz para inglês
            translated = translator.translate(text)
            return translated, True

    except LangDetectException:
        # Não conseguiu detectar o idioma, tenta traduzir mesmo assim
        try:
            translated = translator.translate(text)
            # Se a tradução for muito similar ao original, provavelmente já era inglês
            if translated.lower().strip() == text.lower().strip():
                return text, False
            return translated, True
        except Exception as e:
            st.warning(f"Não foi possível traduzir. Usando texto original.")
            return text, False

    except Exception as e:
        st.warning(f"Erro na tradução: {str(e)}. Usando texto original.")
        return text, False
