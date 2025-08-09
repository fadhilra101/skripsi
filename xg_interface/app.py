"""
xG Prediction Interface - Main Application

A Streamlit application for predicting Expected Goals (xG) from shot data.
Supports both batch prediction from uploaded datasets and individual shot simulation.
"""

import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.model_manager import create_dummy_model_if_not_exists, load_model
from src.pages.dataset_prediction import render_dataset_prediction_page
from src.pages.custom_shot import render_custom_shot_page
from src.utils.language import LANGUAGES, get_translation


def initialize_session_state():
    """Initialize session state variables."""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'  # Default to English


def render_language_switcher():
    """Render the language switcher in the sidebar."""
    st.sidebar.markdown("---")
    
    # Get current language for display
    current_lang = st.session_state.language
    t_lang = get_translation("language", current_lang)
    
    # Language selector
    selected_display = None
    for display, code in LANGUAGES.items():
        if code == current_lang:
            selected_display = display
            break
    
    new_language_display = st.sidebar.selectbox(
        f"🌐 {t_lang}",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(selected_display) if selected_display else 0
    )
    
    # Update language if changed
    new_language_code = LANGUAGES[new_language_display]
    if new_language_code != st.session_state.language:
        st.session_state.language = new_language_code
        st.rerun()


def render_author_info():
    """Render author information and copyright in the sidebar."""
    lang = st.session_state.language
    
    st.sidebar.markdown("---")
    
    # Author info section
    if lang == "id":
        st.sidebar.markdown("""
        ### 👨‍💻 Tentang Pembuat
        
        **Dibuat oleh:**  
        🎓 **Fadhil Raihan Akbar**  
        🏛️ **UIN Syarif Hidayatullah Jakarta**  
        📚 **Program Studi Sistem Informasi**  
        
        📝 **Tugas Akhir Strata 1**  
        *PENERAPAN LIGHT GRADIENT BOOSTING MACHINE (LIGHTGBM) UNTUK PREDIKSI NILAI EXPECTED GOALS (xG) DALAM ANALISIS SEPAK BOLA*
        
        **Kontak & Portfolio:**  
        🔗 [GitHub](https://github.com/fadhilra101)  
        📱 [Instagram](https://www.instagram.com/fadhilra_)
        
        ---
        💡 *Aplikasi ini dikembangkan sebagai bagian dari penelitian tugas akhir untuk membantu analisis performa dalam sepak bola menggunakan teknologi machine learning.*
        """)
    else:
        st.sidebar.markdown("""
        ### 👨‍💻 About the Creator
        
        **Created by:**  
        🎓 **Fadhil Raihan Akbar**  
        🏛️ **UIN Syarif Hidayatullah Jakarta**  
        📚 **Information Systems Study Program**  
        
        📝 **Undergraduate Thesis Project**  
        *APPLICATION OF LIGHT GRADIENT BOOSTING MACHINE (LIGHTGBM) FOR EXPECTED GOALS (xG) VALUE PREDICTION IN FOOTBALL ANALYSIS*
        
        **Contact & Portfolio:**  
        🔗 [GitHub](https://github.com/fadhilra101)  
        📱 [Instagram](https://www.instagram.com/fadhilra_)
        
        ---
        💡 *This application was developed as part of undergraduate thesis research to assist football performance analysis using machine learning technology.*
        """)
    
    # Copyright notice
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 20px;'>
    © 2025 Fadhil Raihan Akbar<br>
    All Rights Reserved
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Get current language
    lang = st.session_state.language
    
    # Configure Streamlit page
    st.set_page_config(
        layout="wide", 
        page_title="xG Prediction App",
        page_icon="⚽"
    )
    
    # App title
    st.title(get_translation("app_title", lang))
    
    # Initialize model
    create_dummy_model_if_not_exists()
    model = load_model()
    
    if model is None:
        st.error(get_translation("model_load_error", lang))
        return
    
    # Sidebar navigation and language switcher
    st.sidebar.title(get_translation("navigation", lang))
    
    # Page selection
    page_options = [
        get_translation("predict_from_dataset", lang),
        get_translation("simulate_custom_shot", lang)
    ]
    
    page = st.sidebar.radio(
        get_translation("navigation", lang), 
        page_options,
        label_visibility="collapsed"
    )
    
    # Language switcher
    render_language_switcher()
    
    # Author information
    render_author_info()
    
    # Route to appropriate page
    if page == get_translation("predict_from_dataset", lang):
        render_dataset_prediction_page(model, lang)
    elif page == get_translation("simulate_custom_shot", lang):
        render_custom_shot_page(model, lang)


if __name__ == "__main__":
    main()
