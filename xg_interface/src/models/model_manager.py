"""
Model management utilities for the xG prediction application.
"""

import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..utils.constants import MODEL_FILE, CATEGORICAL_FEATURES, FEATURE_NAMES
from ..utils.data_processing import preprocess_shot_data


def check_model_exists():
    """
    Checks if the model file exists and returns the full path.
    
    Returns:
        tuple: (exists: bool, full_path: str)
    """
    full_path = os.path.abspath(MODEL_FILE)
    exists = os.path.exists(MODEL_FILE)
    return exists, full_path


def show_model_not_found_message(lang='en'):
    """
    Shows a clear message when the model file is not found,
    with instructions on where to place it.
    
    Args:
        lang: Language code ('en' or 'id')
    """
    from ..utils.language import get_translation
    
    model_exists, full_path = check_model_exists()
    
    if not model_exists:
        st.error("🚫 **" + get_translation("model_not_found_title", lang) + "**")
        st.markdown("---")
        
        # Main instruction
        st.markdown("### 📁 **" + get_translation("required_action", lang) + ":**")
        st.markdown(f"""
        {get_translation("copy_model_instruction", lang)} `xg_model.joblib` {get_translation("to_path", lang)}:
        
        **📍 Path:** `{full_path}`
        """)
        
        # Additional instructions
        st.markdown("### 🔧 **" + get_translation("steps_to_fix", lang) + ":**")
        st.markdown(f"""
        1. **{get_translation("locate_model", lang)}** (`xg_model.joblib`)
        2. **{get_translation("copy_to_root", lang)}** 
        3. **{get_translation("click_refresh", lang)}**
        """)
        
        # Refresh button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(get_translation("check_model_again", lang), type="primary", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        with st.expander("ℹ️ **Model File Requirements**"):
            st.markdown("""
            - **File name:** Must be exactly `xg_model.joblib`
            - **Format:** Joblib-serialized scikit-learn pipeline
            - **Features:** Must support the required input features
            - **Size:** Typically 1-50 MB depending on model complexity
            """)
        
        # Troubleshooting
        with st.expander("🆘 **Troubleshooting**"):
            st.markdown(f"""
            **If you don't have a model file:**
            - Train a model using your data
            - Save it as: `joblib.dump(model, 'xg_model.joblib')`
            - Place it in: `{os.path.dirname(full_path)}`
            
            **If the file exists but not loading:**
            - Check file permissions
            - Ensure it's a valid joblib file
            - Verify the model is compatible
            """)
        
        return False
    return True


def create_dummy_model_if_not_exists(lang='en'):
    """
    Checks if the model file exists. If not, shows instructions
    instead of creating a dummy model.
    
    Args:
        lang: Language code ('en' or 'id')
    """
    return show_model_not_found_message(lang)


@st.cache_resource
def load_model():
    """
    Loads the pre-trained model from the joblib file.
    
    Returns:
        Trained model pipeline or None if loading fails
    """
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_FILE}. Please ensure it's in the correct directory.")
        return None


def safe_load_model():
    """
    Safely loads the model with proper error handling.
    Returns model if successful, None if failed, and sets appropriate session state.
    """
    try:
        if not os.path.exists(MODEL_FILE):
            if 'model_missing_mid_work' not in st.session_state:
                st.session_state.model_missing_mid_work = True
            return None
            
        model = joblib.load(MODEL_FILE)
        # Model loaded successfully, clear any missing flag
        if 'model_missing_mid_work' in st.session_state:
            del st.session_state.model_missing_mid_work
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        if 'model_missing_mid_work' not in st.session_state:
            st.session_state.model_missing_mid_work = True
        return None


def check_model_during_work():
    """
    Check if model is available during workflow.
    Returns True if model exists and is loadable, False otherwise.
    """
    return safe_load_model() is not None


def predict_xg(model, shot_data: pd.DataFrame) -> tuple:
    """
    Predict xG for a single shot or batch of shots with safe error handling and timing.
    
    Args:
        model: Trained model pipeline (can be None if model missing)
        shot_data: Preprocessed shot data DataFrame
        
    Returns:
        tuple: (xG predictions (probability of goal) or None if error/model missing, prediction_time_seconds)
    """
    import time
    
    # Start timing
    start_time = time.perf_counter()
    
    try:
        # Check if model is None (missing during work)
        if model is None:
            st.session_state.model_missing_mid_work = True
            end_time = time.perf_counter()
            return None, end_time - start_time
            
        # Get model's expected features
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_
        else:
            # Fallback to our predefined features
            from ..utils.constants import FEATURE_NAMES
            required_features = FEATURE_NAMES
        
        # Check which features are missing and add them with default values
        for feature in required_features:
            if feature not in shot_data.columns:
                if feature == 'period':
                    shot_data[feature] = 1  # Default to first half
                elif feature == 'shot_first_time':
                    shot_data[feature] = 0  # Default to not first time
                elif feature == 'shot_key_pass':
                    shot_data[feature] = 0  # Default to not from key pass
                elif feature == 'distance_to_goal':
                    # This should be calculated in preprocessing, but add fallback
                    shot_data[feature] = 0
                elif feature == 'angle_to_goal':
                    # This should be calculated in preprocessing, but add fallback  
                    shot_data[feature] = 0
                else:
                    # For any other missing feature, set to 0
                    shot_data[feature] = 0
        
        # Select only the features the model expects
        X_pred = shot_data[required_features]
        
        # Make prediction (this is the actual model performance we want to measure)
        prediction_start = time.perf_counter()
        predictions = model.predict_proba(X_pred)[:, 1]
        prediction_end = time.perf_counter()
        
        # Calculate pure model prediction time
        pure_model_time = prediction_end - prediction_start
        
        return predictions, pure_model_time
        
    except Exception as e:
        end_time = time.perf_counter()
        st.error(f"An error occurred during prediction: {e}")
        return None, end_time - start_time


def show_model_missing_during_work(lang='en'):
    """
    Shows message when model goes missing during work, with session preservation info.
    """
    from ..utils.language import get_translation
    
    st.error("🚫 **" + get_translation("model_missing_during_work", lang) + "**")
    st.markdown("---")
    
    # Preserve work message
    st.info("💾 **" + get_translation("work_preserved", lang) + "**")
    st.markdown(f"""
    {get_translation("dont_worry", lang)}:
    - 🎯 {get_translation("simulation_state_saved", lang)}
    - ⚙️ {get_translation("settings_preserved", lang)}
    """)
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 🔧 **" + get_translation("quick_fix", lang) + ":**")
    model_exists, full_path = check_model_exists()
    st.markdown(f"""
    1. **{get_translation("restore_model", lang)}** `xg_model.joblib` {get_translation("to_path", lang)}: `{full_path}`
    2. **{get_translation("click_continue", lang)}**
    """)
    
    # Continue button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(get_translation("continue_work", lang), type="primary", use_container_width=True):
            # Try to load model and continue
            model = safe_load_model()
            if model is not None:
                st.rerun()
            else:
                st.error(get_translation("model_still_missing", lang))
    
    return False
