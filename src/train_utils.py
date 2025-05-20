import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # Ensure Adam is imported
from . import config

def prepare_data_for_dl(X_features, y_labels, num_classes):
    """ Prepares data for DL: to_categorical and reshaping for Conv1D. """
    y_oh = to_categorical(y_labels, num_classes=num_classes)
    X_reshaped = X_features.reshape((X_features.shape[0], 1, X_features.shape[1]))
    return X_reshaped, y_oh

def prepare_data_for_ml(X_features, y_labels):
    """ Prepares data for ML (no reshaping or one-hot encoding for y here). """
    return X_features, y_labels # y_labels are kept as original indices for sklearn

def split_and_scale_data(X, y, stratify_labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, for_dl=True):
    """ Splits data and scales features. Handles DL reshaping and SMOTE if specified. """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )

    # Store original shapes for potential reshaping later
    original_X_train_shape = X_train.shape
    
    # Scaling requires 2D input
    if X_train.ndim > 2: # (samples, timesteps, features) -> (samples*timesteps, features)
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    else: # Already (samples, features)
        X_train_2d = X_train
        X_test_2d = X_test
        
    scaler = StandardScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # Reshape back if original was > 2D (primarily for DL Conv1D)
    if for_dl and X_train.ndim > 2:
        X_train_scaled = X_train_scaled_2d.reshape(original_X_train_shape)
        X_test_scaled = X_test_scaled_2d.reshape(X_test.shape[0], original_X_train_shape[1], original_X_train_shape[2])
    else: # For ML or if original X was already 2D
        X_train_scaled = X_train_scaled_2d
        X_test_scaled = X_test_scaled_2d
        
    return X_train_scaled, X_test_scaled, y_train, y_test

def apply_smote(X_train, y_train, num_classes):
    """ Applies SMOTE to the training data. Expects X_train as (N, 1, num_features) and y_train one-hot. """
    n_samples_train, _, n_features = X_train.shape
    X_train_2d = X_train.reshape(n_samples_train, n_features) # SMOTE needs 2D X
    
    y_train_indices = np.argmax(y_train, axis=1) # SMOTE needs 1D y
    
    print(f"Shape of X_train before SMOTE: {X_train_2d.shape}")
    print(f"Shape of y_train_indices before SMOTE: {y_train_indices.shape}")
    
    sm = SMOTE(random_state=config.RANDOM_STATE)
    X_train_2d_sm, y_train_idx_sm = sm.fit_resample(X_train_2d, y_train_indices)
    
    y_train_sm_oh = to_categorical(y_train_idx_sm, num_classes=num_classes)
    
    n_samples_sm = X_train_2d_sm.shape[0]
    X_train_sm_reshaped = X_train_2d_sm.reshape((n_samples_sm, 1, n_features)) # Reshape back for Conv1D
    
    print("After SMOTE:")
    print(f"X_train_sm_reshaped shape: {X_train_sm_reshaped.shape}")
    print(f"y_train_sm_oh shape: {y_train_sm_oh.shape}")
    
    return X_train_sm_reshaped, y_train_sm_oh


def train_keras_model(model, X_train, y_train, X_test, y_test, 
                      epochs=config.DL_EPOCHS, 
                      batch_size=config.DL_BATCH_SIZE, 
                      learning_rate=config.DL_LEARNING_RATE_INITIAL):
    """ Compiles and trains a Keras model. """
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=config.ES_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=config.RLR_FACTOR, patience=config.RLR_PATIENCE, 
                          min_lr=1e-6, verbose=1) # Adjusted min_lr based on notebook
    ]
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(X_test, y_test)
    )
    return history

def train_sklearn_model(model, X_train_scaled, y_train):
    """ Trains a scikit-learn model. """
    print(f"\n--- Training {model.__class__.__name__} ---")
    model.fit(X_train_scaled, y_train)
    return model
