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

    # Convert to numpy arrays for consistent handling if not already
    audio_segments_np = np.array(audio_segments, dtype=object) # dtype=object for lists of varying lengths
    labels_numeric_np = np.array(labels_numeric)

    print(f"Total audio segments loaded: {len(audio_segments_np)}")
    print(f"Total labels loaded: {len(labels_numeric_np)}")

    # 3. Split data into training and testing sets BEFORE feature extraction
    print("\nSplitting data into training and testing sets...")
    train_segments, test_segments, train_labels, test_labels = train_test_split(
        audio_segments_np,
        labels_numeric_np,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels_numeric_np
    )
    print(f"Number of training segments: {len(train_segments)}")
    print(f"Number of testing segments: {len(test_segments)}")
    print(f"Number of training labels: {len(train_labels)}")
    print(f"Number of testing labels: {len(test_labels)}")

    # 4. Process features for the training set (with augmentation)
    print("\nProcessing features for training set (with augmentation)...")
    train_features_df = feature_extractor.process_features_for_all_audio(
        list(train_segments), list(train_labels), feature_type="stats_cell5", augment=True
    )
    X_train_features = np.array(train_features_df['feature'].tolist())
    y_train = np.array(train_features_df['class'].tolist()) # y_train_labels_numeric
    
    if X_train_features.size == 0:
        print("No features extracted for the training set. Exiting.")
        return
    print(f"Shape of X_train_features: {X_train_features.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    # 5. Process features for the test set (WITHOUT augmentation)
    print("\nProcessing features for test set (without augmentation)...")
    test_features_df = feature_extractor.process_features_for_all_audio(
        list(test_segments), list(test_labels), feature_type="stats_cell5", augment=False
    )
    X_test_features = np.array(test_features_df['feature'].tolist())
    y_test = np.array(test_features_df['class'].tolist()) # y_test_labels_numeric

    if X_test_features.size == 0:
        print("No features extracted for the test set. Exiting.")
        return
    print(f"Shape of X_test_features: {X_test_features.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    
    num_features_extracted = X_train_features.shape[1]
    print(f"\nNumber of features extracted: {num_features_extracted}")

    # 6. Scale features
    # Initialize a StandardScaler
    # Fit the scaler ONLY on X_train_features and transform both
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features) # Apply the same transformation to the test set
    
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    
    # 7. Get and Train Model
    rf_model = models.get_rf_classifier()
    trained_rf_model = train_utils.train_sklearn_model(rf_model, X_train_scaled, y_train)

    # 8. Evaluate
    evaluate_utils.evaluate_model_performance(
        trained_rf_model, X_test_scaled, y_test,
        model_name="Random Forest (Combined Stats + Augmentation, Split Before Aug)",
        is_keras_model=False
    )

if __name__ == '__main__':
    run_experiment_ml_rf_combined_augmented()
