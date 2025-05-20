import os
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tqdm import tqdm # Use standard tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    LSTM, Dense, Dropout, Bidirectional, Reshape
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from . import config

warnings.filterwarnings("ignore")

# --- Configuration ---
# Data Path
KAGGLE_INPUT_PATH = '/kaggle/input/baby-crying-sounds-datasets/Baby Crying Sounds/'
LOCAL_DATA_PATH = './data/Baby Crying Sounds/'
WORKING_DIR = ""
if os.path.exists(KAGGLE_INPUT_PATH) and os.path.isdir(KAGGLE_INPUT_PATH):
    WORKING_DIR = KAGGLE_INPUT_PATH
    print(f"Using Kaggle directory: {WORKING_DIR}")
elif os.path.exists(LOCAL_DATA_PATH) and os.path.isdir(LOCAL_DATA_PATH):
    WORKING_DIR = LOCAL_DATA_PATH
    print(f"Kaggle directory not found. Using local directory: {WORKING_DIR}")
else:
    print(f"ERROR: Data directory not found at {KAGGLE_INPUT_PATH} or {LOCAL_DATA_PATH}. Please check paths.")
    exit()

CLASS_INDEXES = {
    "belly pain": 0, "burping": 1, "cold_hot": 2, "discomfort": 3,
    "hungry": 4, "laugh": 5, "noise": 6, "silence": 7, "tired": 8
}
CLASS_LABELS_LIST = list(CLASS_INDEXES.keys())
NUM_CLASSES = len(CLASS_INDEXES)

# Audio Processing
SAMPLE_RATE = 22050
DURATION_SECONDS = 5  # Process up to 5 seconds
MAX_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

# Feature Extraction (Sequence-based)
N_MFCC = 20  # Number of MFCC coefficients
HOP_LENGTH = 512
N_FFT = 2048
MAX_PAD_LEN = int(np.ceil(MAX_SAMPLES / HOP_LENGTH)) # Max number of frames/timesteps

# Augmentation
NOISE_FACTOR = 0.005 # Reduced noise factor
TIME_SHIFT_MAX_PERCENT = 0.2 # Shift up to 20% of audio length
PITCH_SHIFT_STEPS = 2 # Shift up to 2 semitones

# Training
TEST_SIZE = 0.2 # Using 20% for test, leaving more for train+SMOTE
RANDOM_STATE = 42
DL_EPOCHS = 200 # Allow more epochs with early stopping
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 0.0005 # Slightly lower learning rate
L2_REG = 1e-4
DROP_RATE = 0.4 # Slightly adjusted dropout

# --- Data Loading and Preprocessing ---

def load_and_process_audio(file_path, sr, max_samples):
    try:
        audio, current_sr = librosa.load(file_path, sr=sr)
        if len(audio) > max_samples: # Truncate
            audio = audio[:max_samples]
        elif len(audio) < max_samples: # Pad
            audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
        
        if len(audio) < N_FFT: # Basic check
            return None
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# --- Augmentation Functions ---
def augment_add_noise(audio_data, noise_factor=NOISE_FACTOR):
    noise = np.random.randn(len(audio_data))
    augmented_audio = audio_data + noise_factor * noise
    return np.clip(augmented_audio, -1., 1.)

def augment_time_shift(audio_data, sr, max_shift_percent=TIME_SHIFT_MAX_PERCENT):
    shift_max_samples = int(len(audio_data) * max_shift_percent)
    shift = np.random.randint(-shift_max_samples, shift_max_samples)
    return np.roll(audio_data, shift)

def augment_pitch_shift(audio_data, sr, n_steps=PITCH_SHIFT_STEPS):
    return librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=n_steps)

# --- Feature Extraction (Sequence) ---
def extract_mfcc_sequence(audio, sr, n_mfcc, hop_length, n_fft, max_pad_len):
    if audio is None or len(audio) < n_fft: return None
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Pad or truncate MFCC sequence
    if mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs.T # Transpose to (timesteps, features)

# --- Main Data Pipeline ---
def get_data(apply_augmentation_to_train=True):
    all_audio_segments = []
    all_labels = []
    
    audio_files_info = []
    sub_dirs_info = [(name, os.path.join(WORKING_DIR, name)) for name in CLASS_INDEXES.keys()]

    for class_name, dir_path in sub_dirs_info:
        if not os.path.isdir(dir_path): continue
        for file_path_obj in pathlib.Path(dir_path).iterdir():
            if file_path_obj.is_file() and file_path_obj.suffix.lower() in ['.wav', '.mp3']:
                audio_files_info.append((CLASS_INDEXES[class_name], file_path_obj))
    
    print(f"Found {len(audio_files_info)} audio files.")

    for class_idx, file_path_obj in tqdm(audio_files_info, desc="Loading audio"):
        audio = load_and_process_audio(str(file_path_obj), SAMPLE_RATE, MAX_SAMPLES)
        if audio is not None:
            all_audio_segments.append(audio)
            all_labels.append(class_idx)
            
    X_features_seq = []
    y_features = []

    print("\nExtracting features...")
    for i, audio_segment in enumerate(tqdm(all_audio_segments, desc="Extracting sequence features")):
        mfcc_seq = extract_mfcc_sequence(audio_segment, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN)
        if mfcc_seq is not None:
            X_features_seq.append(mfcc_seq)
            y_features.append(all_labels[i]) # Corresponding label for original audio

    X_features_seq = np.array(X_features_seq)
    y_features = np.array(y_features)
    
    print(f"X_features_seq shape: {X_features_seq.shape}") # (samples, timesteps, n_mfcc)
    print(f"y_features shape: {y_features.shape}")

    # Train-test split BEFORE augmentation and SMOTE
    X_train_seq, X_test_seq, y_train_orig, y_test_orig = train_test_split(
        X_features_seq, y_features, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y_features
    )

    # Data Augmentation on TRAINING audio segments (if needed for feature re-extraction)
    # For simplicity here, we'll augment the *feature sequences* if apply_augmentation_to_train is True.
    # A more robust way is to augment audio then re-extract features, but this is faster for now.
    # Or, perform augmentation on audio segments before feature extraction loop.
    
    X_train_augmented = list(X_train_seq)
    y_train_augmented_labels = list(y_train_orig)

    if apply_augmentation_to_train:
        print("\nAugmenting training data (based on original audio - conceptual for feature space)...")
        # This part needs to be re-thought if augmenting features directly.
        # The current setup loads audio, then extracts features.
        # To augment effectively for sequence models, we'd augment audio *then* extract features.
        # Let's simulate by adding noise to extracted MFCC sequences for now.
        # This is a simplification.
        temp_audio_segments_train = [] # Need to reload train audio for augmentation
        temp_labels_train = []
        
        # Re-find training file paths (this is inefficient, ideally store paths earlier)
        train_indices, _ = train_test_split(np.arange(len(audio_files_info)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels_numeric if 'labels_numeric' in locals() else y_features)

        print("Reloading training audio for augmentation...")
        for i in tqdm(train_indices, desc="Reloading train audio"):
            class_idx, file_path_obj = audio_files_info[i]
            audio = load_and_process_audio(str(file_path_obj), SAMPLE_RATE, MAX_SAMPLES)
            if audio is not None:
                temp_audio_segments_train.append(audio)
                temp_labels_train.append(class_idx)

        print("\nApplying augmentations to training audio and re-extracting features...")
        for i, audio_segment in enumerate(tqdm(temp_audio_segments_train, desc="Augmenting & Re-extracting")):
            original_label_idx = temp_labels_train[i]
            original_label_name = CLASS_LABELS_LIST[original_label_idx]

            if original_label_name not in config.CLASSES_TO_AUGMENT_LESS: # Example: Don't augment 'hungry' as much
                # Noise
                noisy_audio = augment_add_noise(audio_segment)
                mfcc_seq_noise = extract_mfcc_sequence(noisy_audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN)
                if mfcc_seq_noise is not None:
                    X_train_augmented.append(mfcc_seq_noise)
                    y_train_augmented_labels.append(original_label_idx)
                
                # Time Shift (conceptual, as we work with fixed length here after loading)
                # If features were extracted *after* time shift on raw audio, it would be more effective
                # shifted_audio = augment_time_shift(audio_segment, SAMPLE_RATE)
                # mfcc_seq_shift = extract_mfcc_sequence(shifted_audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN)
                # if mfcc_seq_shift is not None:
                #     X_train_augmented.append(mfcc_seq_shift)
                #     y_train_augmented_labels.append(original_label_idx)

                # Pitch Shift
                pitched_audio = augment_pitch_shift(audio_segment, SAMPLE_RATE)
                mfcc_seq_pitch = extract_mfcc_sequence(pitched_audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN)
                if mfcc_seq_pitch is not None:
                    X_train_augmented.append(mfcc_seq_pitch)
                    y_train_augmented_labels.append(original_label_idx)

        X_train_seq = np.array(X_train_augmented)
        y_train_orig = np.array(y_train_augmented_labels)
        print(f"X_train_seq shape after augmentation: {X_train_seq.shape}")


    # SMOTE on TRAINING data (after feature extraction and augmentation)
    # SMOTE needs 2D input for X, so we reshape, apply SMOTE, then reshape back
    n_samples_train, n_timesteps, n_features_mfcc = X_train_seq.shape
    X_train_2d_for_smote = X_train_seq.reshape(n_samples_train, n_timesteps * n_features_mfcc)
    
    print(f"\nApplying SMOTE to training data...")
    print(f"X_train shape before SMOTE: {X_train_2d_for_smote.shape}")
    print(f"y_train shape before SMOTE: {y_train_orig.shape}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm_2d, y_train_sm = smote.fit_resample(X_train_2d_for_smote, y_train_orig)
    
    # Reshape X_train_sm back to 3D for Conv1D/LSTM
    X_train_sm_3d = X_train_sm_2d.reshape(X_train_sm_2d.shape[0], n_timesteps, n_features_mfcc)
    
    print("After SMOTE:")
    print(f"X_train_sm (3D) shape: {X_train_sm_3d.shape}")
    print(f"y_train_sm shape: {y_train_sm.shape}")
    
    # One-hot encode labels
    y_train_sm_oh = to_categorical(y_train_sm, num_classes=NUM_CLASSES)
    y_test_oh = to_categorical(y_test_orig, num_classes=NUM_CLASSES)
    
    return X_train_sm_3d, X_test_seq, y_train_sm_oh, y_test_oh, y_test_orig # Return y_test_orig for sklearn metrics

# --- Model Definition ---
def build_advanced_dl_model(input_shape, num_classes, l2_reg, drop_rate):
    model = Sequential([
        Input(shape=input_shape), # (timesteps, features) e.g. (MAX_PAD_LEN, N_MFCC)
        
        Conv1D(64, kernel_size=5, padding="same", activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(drop_rate),
        
        Conv1D(128, kernel_size=5, padding="same", activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(drop_rate),

        Conv1D(256, kernel_size=5, padding="same", activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(drop_rate),
        
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))),
        Dropout(drop_rate),
        Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))),
        Dropout(drop_rate),
        
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(drop_rate),
        
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(drop_rate),
        
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=DL_LEARNING_RATE), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Training and Evaluation ---
def plot_history(history_df, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.show()

def main():
    X_train, X_test, y_train_oh, y_test_oh, y_test_indices = get_data(apply_augmentation_to_train=True)

    input_shape_dl = (X_train.shape[1], X_train.shape[2]) # (timesteps, features)
    
    model = build_advanced_dl_model(input_shape_dl, NUM_CLASSES, L2_REG, DROP_RATE)
    model.summary()

    checkpoint_filepath = './best_model_advanced.keras' # Keras v3 format
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1), # Increased patience
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1), # Smaller min_lr
        ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_test, y_test_oh),
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights for final evaluation
    model.load_weights(checkpoint_filepath)

    history_df = pd.DataFrame(history.history)
    plot_history(history_df, "Advanced DL Model")

    loss, accuracy = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\nAdvanced DL Model Test Accuracy: {accuracy:.4f}")
    print(f"Advanced DL Model Test Loss: {loss:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_prob, axis=1)

    # Ensure target_names aligns with the unique sorted labels present in y_test_indices and y_pred_indices
    present_labels_numeric = np.unique(np.concatenate((y_test_indices, y_pred_indices)))
    present_labels_numeric.sort()
    report_target_names = [CLASS_LABELS_LIST[i] for i in present_labels_numeric if i < len(CLASS_LABELS_LIST)]


    print("\nAdvanced DL Model Classification Report:\n", 
          classification_report(y_test_indices, y_pred_indices, 
                                labels=present_labels_numeric, 
                                target_names=report_target_names, 
                                zero_division=0))
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_indices, y_pred_indices, average='macro', labels=present_labels_numeric, zero_division=0
    )
    print(f"Precision (Macro - present labels): {precision:.4f}")
    print(f"Recall (Macro - present labels): {recall:.4f}")
    print(f"F1-Score (Macro - present labels): {f1:.4f}")


    cm = confusion_matrix(y_test_indices, y_pred_indices, labels=present_labels_numeric)
    print("\nConfusion Matrix:\n", cm)
    
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=report_target_names)
        fig, ax = plt.subplots(figsize=(max(8, len(report_target_names)*0.9), max(6, len(report_target_names)*0.8)))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.title('Confusion Matrix - Advanced DL Model')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting CM: {e}")

if __name__ == '__main__':
    main()