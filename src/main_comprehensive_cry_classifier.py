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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # For scaling features if needed (less common for raw LSTM input)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    LSTM, Dense, Dropout, Bidirectional, Reshape,
    Permute, Multiply, Activation, Lambda, AdditiveAttention, Attention
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    LSTM, Dense, Dropout, Bidirectional, Reshape,
    Attention, GlobalAveragePooling1D, Layer # <--- ADD Layer HERE
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

warnings.filterwarnings("ignore")

# --- Configuration ---
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

SAMPLE_RATE = 22050
DURATION_SECONDS = 4 # Reduced slightly to manage sequence length
MAX_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

N_MFCC = 13 # Common number for MFCCs
HOP_LENGTH = 512
N_FFT = 2048
MAX_PAD_LEN_MFCC = int(np.ceil(MAX_SAMPLES / HOP_LENGTH))

NOISE_FACTOR = 0.005
TIME_SHIFT_MAX_PERCENT = 0.15
PITCH_SHIFT_STEPS_MAX = 2 # Max semitones to shift

TEST_SIZE = 0.2
RANDOM_STATE = 42
DL_EPOCHS = 250 # Increased for potentially longer convergence
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 0.0003 # Reduced further
L2_REG = 1e-4
DROP_RATE = 0.45

# --- Data Loading and Preprocessing ---
def load_and_process_audio(file_path, sr, max_samples):
    try:
        audio, current_sr = librosa.load(file_path, sr=sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
        if len(audio) < N_FFT: return None
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
    shift = np.random.randint(-shift_max_samples, shift_max_samples + 1) # Ensure +1 for upper bound
    return np.roll(audio_data, shift)

def augment_pitch_shift(audio_data, sr, n_steps_max=PITCH_SHIFT_STEPS_MAX):
    n_steps = np.random.randint(-n_steps_max, n_steps_max + 1)
    if n_steps == 0: return audio_data # No shift
    return librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=float(n_steps))


# --- Feature Extraction (Sequence) ---
def extract_combined_sequences(audio, sr, n_mfcc, hop_length, n_fft, max_pad_len):
    if audio is None or len(audio) < n_fft: return None
    
    # MFCCs, Deltas, Delta-Deltas
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    mfcc_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0) # Shape (3*n_mfcc, n_frames)

    # Pitch (F0) - using pyin
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr,
                            frame_length=n_fft, hop_length=hop_length)
    f0 = np.nan_to_num(f0) # Replace NaNs (unvoiced frames) with 0
    f0 = f0.reshape(1, -1) # Make it (1, n_frames)

    # RMS Energy
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length) # Shape (1, n_frames)

    # Concatenate all features along the feature axis (axis=0 before transpose)
    combined_features = np.concatenate((mfcc_features, f0, rms), axis=0) # Shape (3*n_mfcc + 1 + 1, n_frames)
    
    # Pad or truncate feature sequence
    if combined_features.shape[1] > max_pad_len:
        combined_features = combined_features[:, :max_pad_len]
    else:
        pad_width = max_pad_len - combined_features.shape[1]
        combined_features = np.pad(combined_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    return combined_features.T # Transpose to (timesteps, features)

# --- Main Data Pipeline ---
def get_data(apply_augmentation_to_train=True):
    audio_files_info = []
    sub_dirs_info = [(name, os.path.join(WORKING_DIR, name)) for name in CLASS_INDEXES.keys()]

    for class_name, dir_path in sub_dirs_info:
        if not os.path.isdir(dir_path): continue
        for file_path_obj in pathlib.Path(dir_path).iterdir():
            if file_path_obj.is_file() and file_path_obj.suffix.lower() in ['.wav', '.mp3']:
                audio_files_info.append((CLASS_INDEXES[class_name], file_path_obj))
    
    print(f"Found {len(audio_files_info)} audio files.")

    # Separate file paths and labels for train/test split
    labels_for_stratify = np.array([info[0] for info in audio_files_info])
    filepaths_with_labels = [(info[1], info[0]) for info in audio_files_info]

    # Split file paths before loading and feature extraction
    train_files_info, test_files_info = train_test_split(
        filepaths_with_labels, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=labels_for_stratify
    )

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print("\nProcessing training data...")
    for file_path_obj, class_idx in tqdm(train_files_info, desc="Train Features"):
        audio = load_and_process_audio(str(file_path_obj), SAMPLE_RATE, MAX_SAMPLES)
        if audio is None: continue

        # Original
        features_seq = extract_combined_sequences(audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN_MFCC)
        if features_seq is not None:
            X_train_list.append(features_seq)
            y_train_list.append(class_idx)

        # Augmentation
        if apply_augmentation_to_train:
            if CLASS_LABELS_LIST[class_idx] not in ["hungry", "silence", "noise", "laugh"]: # Augment less for some classes
                # Noise
                noisy_audio = augment_add_noise(audio)
                features_noise = extract_combined_sequences(noisy_audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN_MFCC)
                if features_noise is not None:
                    X_train_list.append(features_noise)
                    y_train_list.append(class_idx)
                
                # Time Shift
                shifted_audio = augment_time_shift(audio, SAMPLE_RATE)
                features_shift = extract_combined_sequences(shifted_audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN_MFCC)
                if features_shift is not None:
                    X_train_list.append(features_shift)
                    y_train_list.append(class_idx)

                # Pitch Shift
                pitched_audio = augment_pitch_shift(audio, SAMPLE_RATE)
                features_pitch = extract_combined_sequences(pitched_audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN_MFCC)
                if features_pitch is not None:
                    X_train_list.append(features_pitch)
                    y_train_list.append(class_idx)
    
    print("\nProcessing test data...")
    for file_path_obj, class_idx in tqdm(test_files_info, desc="Test Features"):
        audio = load_and_process_audio(str(file_path_obj), SAMPLE_RATE, MAX_SAMPLES)
        if audio is None: continue
        features_seq = extract_combined_sequences(audio, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, MAX_PAD_LEN_MFCC)
        if features_seq is not None:
            X_test_list.append(features_seq)
            y_test_list.append(class_idx)

    X_train_seq = np.array(X_train_list)
    y_train_orig = np.array(y_train_list)
    X_test_seq = np.array(X_test_list)
    y_test_orig = np.array(y_test_list)
    
    print(f"\nOriginal X_train_seq shape: {X_train_seq.shape}")
    print(f"Original y_train_orig shape: {y_train_orig.shape}")

    # SMOTE on TRAINING data
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
class ReduceMeanLayer(Layer):
    def __init__(self, axis, keepdims, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)

    def get_config(self): # Important for model saving/loading
        config = super(ReduceMeanLayer, self).get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config


def build_advanced_dl_model_with_attention(input_shape, num_classes, l2_reg, drop_rate):
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
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)))(x) # return_sequences=True for Attention
    x = Dropout(drop_rate)(lstm_out)

    # Attention Layer (using Keras's built-in AdditiveAttention)
    # The query for AdditiveAttention is typically a learned representation or the last state.
    # Here, we'll use a simple approach where the query is a dense transformation of the mean of LSTM outputs.
    
    # --- MODIFIED PART ---
    # query_input = tf.reduce_mean(x, axis=1, keepdims=True) # OLD - This caused the error
    query_input = ReduceMeanLayer(axis=1, keepdims=True, name='reduce_mean_for_query')(x) # NEW - Using custom layer
    query = Dense(128, activation='tanh', name='attention_query')(query_input) # (batch, 1, dim)
    # --- END MODIFIED PART ---
    
    attention_layer = Attention(use_scale=True, dropout=0.1) # AdditiveAttention or standard Attention
    context_vector = attention_layer([query, x, x]) # Query, Value, Key (Value and Key are same here)
    # The output of Attention is (batch, query_seq_len, value_dim). Here query_seq_len is 1.
    # So, context_vector shape is (batch, 1, value_dim_of_lstm_output*2_for_bidirectional)
    # We need to flatten it or take the single time step
    context_vector = Reshape((-1,))(context_vector) # Flatten the context vector output (batch, features)

    x = Dropout(drop_rate)(context_vector) # Moved dropout after flattening attention output
    
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
def plot_history(history_df, model_name):
    # ... (same as before)
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

    if X_train.size == 0 or X_test.size == 0:
        print("Training or testing data is empty. Exiting.")
        return

    input_shape_dl = (X_train.shape[1], X_train.shape[2]) # (timesteps, features)
    
    model = build_advanced_dl_model_with_attention(input_shape_dl, NUM_CLASSES, L2_REG, DROP_RATE)
    model.summary()

    checkpoint_filepath = './best_model_comprehensive.keras'
    
    # Calculate class weights for the SMOTE'd training data
    y_train_sm_indices = np.argmax(y_train_oh, axis=1) 
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_sm_indices),
        y=y_train_sm_indices
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print(f"\nUsing class weights for training: {class_weights_dict}\n")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1), # Increased patience
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-7, verbose=1), # Increased patience
        ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_test, y_test_oh),
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights_dict, # Apply class weights
        verbose=1
    )
    
    model.load_weights(checkpoint_filepath) # Load best weights

    history_df = pd.DataFrame(history.history)
    plot_history(history_df, "Comprehensive DL Model")

    loss, accuracy = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\nComprehensive DL Model Test Accuracy: {accuracy:.4f}")
    print(f"Comprehensive DL Model Test Loss: {loss:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_prob, axis=1)
    
    present_labels_numeric = np.unique(np.concatenate((y_test_indices, y_pred_indices)))
    present_labels_numeric.sort()
    report_target_names = [CLASS_LABELS_LIST[i] for i in present_labels_numeric if i < len(CLASS_LABELS_LIST)]

    print("\nComprehensive DL Model Classification Report:\n", 
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
        plt.title('Confusion Matrix - Comprehensive DL Model')
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"Error plotting CM: {e}")

if __name__ == '__main__':
    main()