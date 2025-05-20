import os
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    LSTM, Dense, Dropout, Bidirectional,
    Attention, GlobalAveragePooling1D 
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# --- Configuration ---
KAGGLE_INPUT_PATH = '/kaggle/input/baby-crying-sounds-datasets/Baby Crying Sounds/'
LOCAL_DATA_PATH = './data/Baby Crying Sounds/' # Ensure this path is correct for your local setup
WORKING_DIR = ""
if os.path.exists(KAGGLE_INPUT_PATH) and os.path.isdir(KAGGLE_INPUT_PATH):
    WORKING_DIR = KAGGLE_INPUT_PATH
    print(f"Using Kaggle directory: {WORKING_DIR}")
elif os.path.exists(LOCAL_DATA_PATH) and os.path.isdir(LOCAL_DATA_PATH):
    WORKING_DIR = LOCAL_DATA_PATH
    print(f"Kaggle directory not found. Using local directory: {WORKING_DIR}")
else:
    print(f"ERROR: Data directory not found at {KAGGLE_INPUT_PATH} or {LOCAL_DATA_PATH}.")
    exit()

CLASS_INDEXES = {
    "belly pain": 0, "burping": 1, "cold_hot": 2, "discomfort": 3,
    "hungry": 4, "laugh": 5, "noise": 6, "silence": 7, "tired": 8
}
CLASS_LABELS_LIST = list(CLASS_INDEXES.keys())
NUM_CLASSES = len(CLASS_INDEXES)

SAMPLE_RATE = 22050
DURATION_SECONDS = 4 
MAX_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

N_MFCC = 20 # Increased MFCCs
HOP_LENGTH = 512
N_FFT = 2048
MAX_SEQ_LEN = int(np.ceil(MAX_SAMPLES / HOP_LENGTH)) # Max number of frames/timesteps

# Augmentation
NOISE_FACTOR_RANGE = (0.001, 0.005) # Range for noise
TIME_SHIFT_MAX_PERCENT = 0.20
PITCH_SHIFT_STEPS_MAX = 3
NUM_AUGMENTATIONS_PER_SAMPLE = 2 # Create N augmented versions for eligible samples

# Training
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1 # Create a separate validation set from training data
RANDOM_STATE = 42
DL_EPOCHS = 300 
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 0.0001 # Further reduced initial LR
L2_REG = 1e-5 # Slightly reduced L2
DROP_RATE = 0.5

# --- Helper Functions ---
def load_and_process_audio(file_path, sr, max_samples):
    try:
        audio, current_sr = librosa.load(file_path, sr=sr)
        if len(audio) > max_samples: audio = audio[:max_samples]
        elif len(audio) < max_samples: audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
        if len(audio) < N_FFT: return None
        return audio
    except Exception as e: return None

def augment_data(audio_data, sr):
    augmented_variations = []
    # 1. Noise
    noise_factor = random.uniform(NOISE_FACTOR_RANGE[0], NOISE_FACTOR_RANGE[1])
    noisy_audio = audio_data + noise_factor * np.random.randn(len(audio_data))
    augmented_variations.append(np.clip(noisy_audio, -1., 1.))
    
    # 2. Time Shift
    shift_max_samples = int(len(audio_data) * TIME_SHIFT_MAX_PERCENT)
    shift = np.random.randint(-shift_max_samples, shift_max_samples + 1)
    shifted_audio = np.roll(audio_data, shift)
    augmented_variations.append(shifted_audio)

    # 3. Pitch Shift
    n_steps = float(np.random.randint(-PITCH_SHIFT_STEPS_MAX, PITCH_SHIFT_STEPS_MAX + 1))
    if n_steps != 0: # Avoid no-op pitch shift if n_steps is 0
        pitched_audio = librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=n_steps)
        augmented_variations.append(pitched_audio)
    else: # If n_steps is 0, maybe add another noise version or a time-stretched version
        stretched_audio = librosa.effects.time_stretch(y=audio_data, rate=random.uniform(0.8, 1.2))
        if len(stretched_audio) > MAX_SAMPLES: stretched_audio = stretched_audio[:MAX_SAMPLES]
        else: stretched_audio = np.pad(stretched_audio, (0, MAX_SAMPLES - len(stretched_audio)), mode='constant')
        augmented_variations.append(stretched_audio)
        
    return augmented_variations


def extract_comprehensive_features(audio, sr, n_mfcc, hop_length, n_fft, max_seq_len):
    if audio is None or len(audio) < n_fft: return None
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr,
                            frame_length=n_fft, hop_length=hop_length)
    f0 = np.nan_to_num(f0).reshape(1, -1)
    
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
    
    hnr = librosa.effects.harmonic(audio, margin=3.0) # margin to separate harmonics from percussive
    # Calculate RMS for harmonic and percussive components to derive a per-frame HNR-like feature
    rms_harmonic = librosa.feature.rms(y=hnr, frame_length=n_fft, hop_length=hop_length)
    rms_percussive = librosa.feature.rms(y=librosa.effects.percussive(audio, margin=3.0), frame_length=n_fft, hop_length=hop_length)
    # Simple HNR-proxy: ratio of harmonic energy to total (or harmonic to percussive)
    # Avoid division by zero
    hnr_proxy = rms_harmonic / (rms_percussive + 1e-6) 


    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)

    features_list = [mfccs, delta_mfccs, delta2_mfccs, f0, rms, hnr_proxy, spectral_centroid, spectral_bandwidth, spectral_rolloff]
    
    # Ensure all features have the same number of frames before concatenating
    min_frames = min(f.shape[1] for f in features_list)
    features_aligned = [f[:, :min_frames] for f in features_list]
    
    combined_features = np.vstack(features_aligned)
    
    if combined_features.shape[1] > max_seq_len:
        combined_features = combined_features[:, :max_seq_len]
    else:
        pad_width = max_seq_len - combined_features.shape[1]
        combined_features = np.pad(combined_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    return combined_features.T

# --- Data Pipeline ---
def get_data_pipeline(augment_train=True, num_augmentations_per_sample=NUM_AUGMENTATIONS_PER_SAMPLE):
    audio_files_info = []
    # ... (same file discovery as before) ...
    sub_dirs_info = [(name, os.path.join(WORKING_DIR, name)) for name in CLASS_INDEXES.keys()]
    for class_name, dir_path in sub_dirs_info:
        if not os.path.isdir(dir_path): continue
        for file_path_obj in pathlib.Path(dir_path).iterdir():
            if file_path_obj.is_file() and file_path_obj.suffix.lower() in ['.wav', '.mp3']:
                audio_files_info.append((CLASS_INDEXES[class_name], str(file_path_obj))) # Store path as string
    
    print(f"Found {len(audio_files_info)} audio files.")
    
    labels_for_stratify = np.array([info[0] for info in audio_files_info])
    
    train_files_info, test_files_info = train_test_split(
        audio_files_info, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=labels_for_stratify
    )

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print("\nProcessing training data...")
    for class_idx, file_path in tqdm(train_files_info, desc="Train Data Processing"):
        audio = load_and_process_audio(file_path, SAMPLE_RATE, MAX_SAMPLES)
        if audio is None: continue

        features_original = extract_comprehensive_features(audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_SEQ_LEN)
        if features_original is not None:
            X_train_list.append(features_original)
            y_train_list.append(class_idx)

        if augment_train:
            original_label_name = CLASS_LABELS_LIST[class_idx]
            # Augment more for problematic classes, less for well-performing or majority ones
            # This is a heuristic and can be fine-tuned
            num_augs_to_apply = num_augmentations_per_sample
            if original_label_name in ["hungry", "laugh", "noise", "silence"]:
                 num_augs_to_apply = 1 # Less augmentation for these

            augmented_audios = []
            if "noise" in original_label_name.lower(): # Don't add noise to noise samples
                 pass
            else:
                augmented_audios.extend(augment_data(audio, SAMPLE_RATE)[:num_augs_to_apply]) # Take a subset of augmentations
            
            for aug_audio in augmented_audios:
                features_aug = extract_comprehensive_features(aug_audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_SEQ_LEN)
                if features_aug is not None:
                    X_train_list.append(features_aug)
                    y_train_list.append(class_idx)
    
    print("\nProcessing test data...")
    for class_idx, file_path in tqdm(test_files_info, desc="Test Data Processing"):
        audio = load_and_process_audio(file_path, SAMPLE_RATE, MAX_SAMPLES)
        if audio is None: continue
        features_seq = extract_comprehensive_features(audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_SEQ_LEN)
        if features_seq is not None:
            X_test_list.append(features_seq)
            y_test_list.append(class_idx)

    X_train_seq = np.array(X_train_list)
    y_train_orig = np.array(y_train_list)
    X_test_seq = np.array(X_test_list)
    y_test_orig = np.array(y_test_list)

    if X_train_seq.size == 0:
        raise ValueError("No training features extracted. Check data paths and processing.")
    if X_test_seq.size == 0:
        raise ValueError("No test features extracted. Check data paths and processing.")

    print(f"\nShape of X_train_seq after augmentation: {X_train_seq.shape}")
    print(f"Shape of y_train_orig after augmentation: {y_train_orig.shape}")
    
    # SMOTE
    n_samples_train, n_timesteps, n_features = X_train_seq.shape
    X_train_2d_for_smote = X_train_seq.reshape(n_samples_train, n_timesteps * n_features)
    
    print(f"\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm_2d, y_train_sm = smote.fit_resample(X_train_2d_for_smote, y_train_orig)
    X_train_sm_3d = X_train_sm_2d.reshape(X_train_sm_2d.shape[0], n_timesteps, n_features)
    
    print("After SMOTE:")
    print(f"X_train_sm_3d shape: {X_train_sm_3d.shape}")
    print(f"y_train_sm shape: {y_train_sm.shape}")
    
    y_train_sm_oh = to_categorical(y_train_sm, num_classes=NUM_CLASSES)
    y_test_oh = to_categorical(y_test_orig, num_classes=NUM_CLASSES)
    
    return X_train_sm_3d, X_test_seq, y_train_sm_oh, y_test_oh, y_test_orig

# --- Model Definition ---
def build_final_dl_model(input_shape, num_classes, l2_reg, drop_rate):
    inputs = Input(shape=input_shape)
    
    # Conv Block 1
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(drop_rate)(x)
    
    # Conv Block 2
    x = Conv1D(128, kernel_size=5, padding="same", activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(drop_rate)(x)

    # Conv Block 3
    x = Conv1D(256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(l2_reg))(x) # Smaller kernel
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(drop_rate)(x)
    
    # BiLSTM Layers
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)))(x)
    x = Dropout(drop_rate)(x)
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)))(x)
    x = Dropout(drop_rate)(lstm_out)

    # Attention Layer
    attention_output = Attention(use_scale=True, dropout=0.1)([x, x]) # Self-attention: query, value, key are all x
    context_vector = GlobalAveragePooling1D()(attention_output) 
    
    x = Dropout(drop_rate)(context_vector)
    
    # Dense Layers
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x) # Increased units
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=DL_LEARNING_RATE)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    return model

# --- Training and Evaluation ---
def plot_history_extended(history_df, model_name):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 3, 3)
    if 'recall' in history_df.columns and 'val_recall' in history_df.columns: # Check if recall metrics exist
        plt.plot(history_df['recall'], label='Train Recall')
        plt.plot(history_df['val_recall'], label='Validation Recall')
        plt.title(f'{model_name} Recall')
        plt.xlabel('Epochs'); plt.ylabel('Recall'); plt.legend()
    
    plt.tight_layout(); plt.show()

def main():
    X_train, X_test, y_train_oh, y_test_oh, y_test_indices = get_data_pipeline(
        augment_train=True, 
        num_augmentations_per_sample=NUM_AUGMENTATIONS_PER_SAMPLE
    )

    if X_train.size == 0 or X_test.size == 0:
        print("Training or testing data is empty after processing. Exiting.")
        return

    input_shape_dl = (X_train.shape[1], X_train.shape[2])
    
    model = build_final_dl_model(input_shape_dl, NUM_CLASSES, L2_REG, DROP_RATE)
    model.summary()

    checkpoint_filepath = './best_model_uber_cry.keras'
    
    y_train_sm_indices = np.argmax(y_train_oh, axis=1) 
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_sm_indices),
        y=y_train_sm_indices
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print(f"\nUsing class weights for training: {class_weights_dict}\n")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1), # More patience
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1), # More patience for LR
        ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_test, y_test_oh),
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    if os.path.exists(checkpoint_filepath):
        print(f"Loading best model weights from {checkpoint_filepath}")
        model.load_weights(checkpoint_filepath)
    else:
        print("Warning: Best model checkpoint not found. Using last epoch weights.")


    history_df = pd.DataFrame(history.history)
    plot_history_extended(history_df, "Uber Comprehensive DL Model")

    loss, accuracy, recall_metric, precision_metric = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\nUber DL Model Test Accuracy: {accuracy:.4f}")
    print(f"Uber DL Model Test Loss: {loss:.4f}")
    print(f"Uber DL Model Test Recall (metric): {recall_metric:.4f}")
    print(f"Uber DL Model Test Precision (metric): {precision_metric:.4f}")


    y_pred_prob = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_prob, axis=1)
    
    present_labels_numeric = np.unique(np.concatenate((y_test_indices, y_pred_indices)))
    present_labels_numeric.sort()
    report_target_names = [CLASS_LABELS_LIST[i] for i in present_labels_numeric if i < len(CLASS_LABELS_LIST)]

    print("\nUber DL Model Classification Report:\n", 
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
        plt.title('Confusion Matrix - Uber Comprehensive DL Model')
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"Error plotting CM: {e}")

if __name__ == '__main__':
    main()