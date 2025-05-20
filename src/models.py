import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from . import config

# --- Keras Models ---

def build_conv1d_lstm_initial(num_features, num_classes):
    """ Conv1D-LSTM from Cell 3/6 """
    model = Sequential([
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", 
                      input_shape=(1, num_features)),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Reshape((1, 256)), # Output of last BN is (None, 1, 256)
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_ffn_cell4(num_features, num_classes):
    """ FFN from Cell 4 """
    model = Sequential([
        layers.Dense(64, activation="relu", input_shape=(num_features,)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def build_conv1d_lstm_smote_regularized(num_features, num_classes):
    """ Conv1D-LSTM with SMOTE and Regularization from Cell 7 """
    l2_reg = config.L2_REG_SMOTE
    drop_rate = config.DROP_RATE_SMOTE
    
    model = Sequential([
        layers.Input(shape=(1, num_features)),
        layers.Conv1D(32, kernel_size=3, padding="same", activation="relu", 
                      kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Reshape((1, 64)),
        layers.LSTM(32, return_sequences=False,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      recurrent_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(drop_rate),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(drop_rate),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg))
    ])
    return model

# --- Scikit-learn Models ---

def get_rf_classifier():
    return RandomForestClassifier(n_estimators=config.RF_N_ESTIMATORS, 
                                  random_state=config.RANDOM_STATE, 
                                  class_weight='balanced', n_jobs=-1)

def get_svm_classifier():
    return SVC(kernel='rbf', C=config.SVM_C, gamma=config.SVM_GAMMA, 
               probability=True, random_state=config.RANDOM_STATE, class_weight='balanced')

def get_knn_classifier():
    return KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS, n_jobs=-1)

def get_bagging_classifier(base_estimator=None):
    if base_estimator is None:
        base_estimator = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS) # Default base
    return BaggingClassifier(estimator=base_estimator, n_estimators=50, 
                             random_state=config.RANDOM_STATE, n_jobs=-1)
