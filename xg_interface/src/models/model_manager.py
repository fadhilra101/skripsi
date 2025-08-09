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


def create_dummy_model_if_not_exists():
    """
    Checks if the model file exists. If not, it creates and trains a dummy
    scikit-learn pipeline and saves it to disk. This ensures the app is
    runnable without needing a pre-existing model file.
    """
    if not os.path.exists(MODEL_FILE):
        st.warning(f"'{MODEL_FILE}' not found. Creating a dummy model for demonstration.")

        # 1. Create a sample DataFrame that mimics the required input structure
        # Boolean features are now 0/1 integers.
        data = {
            'minute': [10, 25, 48, 65, 88],
            'second': [15, 30, 5, 40, 55],
            'period': [1, 1, 2, 2, 2],  # First/Second half
            'play_pattern': [1, 2, 1, 3, 1],  # Regular, Corner, Regular, FK, Regular
            'position': [23, 19, 23, 25, 17], # Striker, CAM, Striker, SS, RW
            'shot_technique': [93, 95, 93, 91, 93], # Normal, Volley, Normal, HV, Normal
            'shot_body_part': [40, 38, 37, 40, 40], # RF, LF, Head, RF, RF
            'shot_type': [87, 61, 87, 62, 87], # Open Play, Corner, OP, FK, OP
            'shot_open_goal': [0, 0, 1, 0, 0],
            'shot_one_on_one': [0, 0, 1, 0, 1],
            'shot_aerial_won': [0, 0, 0, 1, 0],
            'shot_first_time': [0, 1, 0, 0, 1],  # First time shot
            'shot_key_pass': [0, 0, 1, 0, 1],    # Shot from key pass
            'under_pressure': [1, 1, 0, 1, 0],
            'start_x': [105, 110, 115, 95, 112],
            'start_y': [45, 35, 40, 55, 30],
            'type_before': [30, 14, 43, 30, 14], # Pass, Dribble, Carry, Pass, Dribble
            'is_goal': [0, 1, 1, 0, 1] # Target variable
        }
        dummy_df = pd.DataFrame(data)

        # 2. Apply the same feature engineering and preprocessing
        dummy_df = preprocess_shot_data(dummy_df)

        # 3. Define features and target for the model
        X = dummy_df[FEATURE_NAMES]
        y = dummy_df['is_goal']

        # 4. Create a scikit-learn pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)],
            remainder='passthrough')

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])

        # 5. Train the pipeline
        pipeline.fit(X, y)

        # 6. Save the trained pipeline
        joblib.dump(pipeline, MODEL_FILE)
        st.success(f"Dummy model created and saved as '{MODEL_FILE}'.")


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


def predict_xg(model, shot_data: pd.DataFrame) -> float:
    """
    Predict xG for a single shot or batch of shots.
    
    Args:
        model: Trained model pipeline
        shot_data: Preprocessed shot data DataFrame
        
    Returns:
        xG predictions (probability of goal)
    """
    try:
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
        
        # Make prediction
        predictions = model.predict_proba(X_pred)[:, 1]
        return predictions
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None
