import os

# --- Data Configuration ---
# Try Kaggle path first, then a local fallback
KAGGLE_INPUT_PATH = '/kaggle/input/baby-crying-sounds-datasets/Baby Crying Sounds/'
LOCAL_DATA_PATH = './data/Baby Crying Sounds/' # Adjust if your local structure differs

WORKING_DIR = ""
if os.path.exists(KAGGLE_INPUT_PATH):
    WORKING_DIR = KAGGLE_INPUT_PATH
    print(f"Using Kaggle directory: {WORKING_DIR}")
elif os.path.exists(LOCAL_DATA_PATH):
    WORKING_DIR = LOCAL_DATA_PATH
    print(f"Kaggle directory not found. Using local directory: {WORKING_DIR}")
else:
    print(f"ERROR: Data directory not found at {KAGGLE_INPUT_PATH} or {LOCAL_DATA_PATH}. Please check paths.")
    # exit() # Or raise an error

CLASS_INDEXES = {
    "belly pain": 0,
    "burping": 1,
    "cold_hot": 2,
    "discomfort": 3,
    "hungry": 4,
    "laugh": 5,
    "noise": 6,
    "silence": 7,
    "tired": 8
}
CLASS_LABELS_LIST = list(CLASS_INDEXES.keys())
NUM_CLASSES = len(CLASS_INDEXES)

# --- Audio Processing Parameters ---
SAMPLE_RATE = 22050
DURATION_SECONDS = 5  # For fixed-length segments if desired, else None
FIXED_DURATION_PROCESSING = False # Set to True to pad/truncate audio to DURATION_SECONDS

# --- Feature Extraction Parameters ---
# For MFCCs based on Cell 2/3/6/7 (50 MFCCs)
N_MFCC_CELL2 = 50

# For features based on Cell 4 (20 MFCCs + ZCR + RMS)
N_MFCC_CELL4 = 20

# For features based on Cell 5 (Stats of MFCCs, ZCR, RMS)
N_MFCC_CELL5 = 20 # Base MFCCs before stats

# Common librosa params (can be tuned)
HOP_LENGTH = 512
N_FFT = 2048

# --- Augmentation Parameters ---
NOISE_FACTOR = 0.01
CLASSES_TO_AUGMENT_LESS = ["hungry"] # Classes to skip or apply less augmentation

# --- Model Training Parameters ---
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Keras Model Parameters
DL_EPOCHS = 150
DL_BATCH_SIZE = 32
DL_LEARNING_RATE_SMOTE = 0.01 # As used in Cell 7
DL_LEARNING_RATE_INITIAL = 0.001 # As used in Cell 3/6
L2_REG_SMOTE = 1e-4
DROP_RATE_SMOTE = 0.5

# Callbacks
ES_PATIENCE = 20 # For EarlyStopping
RLR_PATIENCE = 10 # For ReduceLROnPlateau
RLR_FACTOR = 0.2 # For ReduceLROnPlateau

# --- ML Model Parameters ---
# (Can add specific params for RF, SVM, KNN if needed)
RF_N_ESTIMATORS = 200
SVM_C = 10
SVM_GAMMA = 'scale'
KNN_N_NEIGHBORS = 5

# --- Experiment Selection (for main scripts) ---
# Example: 'mfcc_cell2_lstm', 'mfcc_cell4_ffn', 'combined_stats_cell5_rf', 'mfcc_cell2_lstm_smote'
CURRENT_EXPERIMENT = 'mfcc_cell2_lstm_smote' # To guide main script execution
