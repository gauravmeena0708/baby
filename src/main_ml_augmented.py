import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from . import data_loader
from . import feature_extractor
from . import models # For scikit-learn model getters
from . import train_utils
from . import evaluate_utils

def run_experiment_ml_rf_combined_augmented():
    print("=== Running ML Experiment (Random Forest, Combined Stats Features, Augmented) ===")

    # 1. Load audio file paths and labels
    audio_files_info = data_loader.get_audio_files(config.WORKING_DIR, config.CLASS_INDEXES)
    if not audio_files_info:
        print("No audio files found. Exiting.")
        return

    # 2. Load audio segments
    audio_segments, labels_numeric = data_loader.load_all_audio_data(audio_files_info)
    
    if not audio_segments:
        print("No audio segments loaded after processing. Exiting.")
        return

    # 3. Extract features (Using combined stats, with augmentation enabled)
    # Augmentation happens *during* feature processing in this setup
    features_df = feature_extractor.process_features_for_all_audio(
        audio_segments, labels_numeric, feature_type="stats_cell5", augment=True
    )
    
    X_features = np.array(features_df['feature'].tolist())
    y_labels_numeric = np.array(features_df['class'].tolist())

    num_features_extracted = X_features.shape[1]
    print(f"Number of features for ML model: {num_features_extracted}")

    # 4. Prepare data for ML (no one-hot for y)
    X_ml, y_ml = train_utils.prepare_data_for_ml(X_features, y_labels_numeric)

    # 5. Train/Test Split & Scale
    # Stratify on y_ml (original numeric labels)
    X_train_scaled, X_test_scaled, y_train, y_test = train_utils.split_and_scale_data(
        X_ml, y_ml, stratify_labels=y_ml, for_dl=False # for_dl=False for sklearn
    )
    print(f"X_train_scaled shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}, y_test shape: {y_test.shape}")
    
    # 6. Get and Train Model
    rf_model = models.get_rf_classifier()
    trained_rf_model = train_utils.train_sklearn_model(rf_model, X_train_scaled, y_train)

    # 7. Evaluate
    evaluate_utils.evaluate_model_performance(
        trained_rf_model, X_test_scaled, y_test, 
        model_name="Random Forest (Combined Stats + Augmentation)", 
        is_keras_model=False
    )

if __name__ == '__main__':
    run_experiment_ml_rf_combined_augmented()
