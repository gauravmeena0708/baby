import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tqdm
from . import config
from . import data_loader
from . import feature_extractor
from . import models
from . import train_utils
from . import evaluate_utils

def run_experiment_dl_smote():
    print("=== Running DL Experiment with SMOTE (Conv1D-LSTM, MFCC_Cell2 features) ===")

    # 1. Load audio file paths and labels
    audio_files_info = data_loader.get_audio_files(config.WORKING_DIR, config.CLASS_INDEXES)
    if not audio_files_info:
        print("No audio files found. Exiting.")
        return

    # 2. Load audio segments
    # For this specific experiment (Cell 7), augmentation was NOT explicitly applied before SMOTE
    # SMOTE itself is a form of synthetic data generation for the minority classes.
    # If you wanted to add noise augmentation *before* SMOTE, you'd do it here on audio_segments.
    audio_segments, labels_numeric = data_loader.load_all_audio_data(audio_files_info)
    
    if not audio_segments:
        print("No audio segments loaded after processing. Exiting.")
        return

    # 3. Extract features (Using MFCC_Cell2 which yields 50 features)
    # No further augmentation specified in Cell 7 for the features themselves, SMOTE is on data points
    features_df = feature_extractor.process_features_for_all_audio(
        audio_segments, labels_numeric, feature_type="mfcc_cell2", augment=False
    )
    
    X_features = np.array(features_df['feature'].tolist())
    y_labels_numeric = np.array(features_df['class'].tolist()) # Original numeric labels

    # 4. Prepare data for DL (reshaping X, one-hot encoding y)
    X_dl, y_dl_oh = train_utils.prepare_data_for_dl(X_features, y_labels_numeric, config.NUM_CLASSES)
    
    num_features_extracted = X_dl.shape[2]
    print(f"Number of features for DL model: {num_features_extracted}")

    # 5. Train/Test Split (stratify on original numeric labels)
    X_train, X_test, y_train_oh, y_test_oh = train_test_split(
        X_dl, y_dl_oh, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y_labels_numeric # Stratify based on original numeric labels before one-hot
    )
    print(f"X_train shape: {X_train.shape}, y_train_oh shape: {y_train_oh.shape}")
    print(f"X_test shape: {X_test.shape}, y_test_oh shape: {y_test_oh.shape}")

    # 6. Apply SMOTE to the training data
    X_train_sm, y_train_sm_oh = train_utils.apply_smote(X_train, y_train_oh, config.NUM_CLASSES)

    # 7. Build and Train Model
    model = models.build_conv1d_lstm_smote_regularized(num_features_extracted, config.NUM_CLASSES)
    model.summary()

    history = train_utils.train_keras_model(
        model, X_train_sm, y_train_sm_oh, X_test, y_test_oh,
        learning_rate=config.DL_LEARNING_RATE_SMOTE # Use specific LR for this model
    )

    # 8. Evaluate
    if history:
        history_df = pd.DataFrame(history.history)
        evaluate_utils.plot_training_history(history_df, "Conv1D-LSTM (SMOTE + Regularized)")
    
    evaluate_utils.evaluate_model_performance(
        model, X_test, y_test_oh, 
        model_name="Conv1D-LSTM (SMOTE + Regularized)", 
        is_keras_model=True
    )

if __name__ == '__main__':
    # This allows running this script directly
    # Ensure PYTHONPATH includes the parent directory of 'src' if running from outside 'src'
    # Example: export PYTHONPATH=$PYTHONPATH:/path/to/baby-cry-classification
    run_experiment_dl_smote()
