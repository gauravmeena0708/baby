import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)
from . import config

def plot_training_history(history_df, model_name="Model"):
    """ Plots loss and accuracy curves from Keras training history. """
    best_epoch = history_df['val_accuracy'].idxmax()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Training loss')
    plt.plot(history_df['val_loss'], label='Validation loss')
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Val Acc Epoch ({best_epoch+1})')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Val Acc Epoch ({best_epoch+1})')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy Curves')
    
    plt.tight_layout()
    plt.show()

def evaluate_model_performance(model, X_test, y_test_oh_or_indices, model_name="Model", is_keras_model=True):
    """
    Evaluates model performance and prints/plots metrics.
    y_test_oh_or_indices: one-hot encoded for Keras, original indices for sklearn.
    """
    if is_keras_model:
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test_oh_or_indices, axis=1)
        loss, accuracy = model.evaluate(X_test, y_test_oh_or_indices, verbose=0)
        print(f"\n{model_name} Test Accuracy: {accuracy:.4f}")
        print(f"{model_name} Test Loss: {loss:.4f}")
    else: # Scikit-learn model
        y_pred = model.predict(X_test)
        y_true = y_test_oh_or_indices # Already indices
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n{model_name} Accuracy: {accuracy:.4f}")

    # Calculate metrics based ONLY on the labels present in y_true and y_pred
    present_labels_numeric = np.unique(np.concatenate((y_true, y_pred)))
    present_labels_numeric.sort()
    
    # Map numeric labels to string names for reporting, only for those present
    report_target_names = [config.CLASS_LABELS_LIST[i] for i in present_labels_numeric if i < len(config.CLASS_LABELS_LIST)]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=present_labels_numeric, zero_division=0
    )
    
    print(f"{model_name} Precision (Macro - based on present labels): {precision:.4f}")
    print(f"{model_name} Recall (Macro - based on present labels): {recall:.4f}")
    print(f"{model_name} F1-Score (Macro - based on present labels): {f1:.4f}")
    
    print(f"\n{model_name} Classification Report:\n", 
          classification_report(y_true, y_pred, labels=present_labels_numeric, target_names=report_target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=present_labels_numeric) # Use present_labels_numeric for CM
    print(f"\n{model_name} Confusion Matrix:\n", cm)
    
    try:
        # Use report_target_names for display_labels in ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=report_target_names)
        fig, ax = plt.subplots(figsize=(max(8, len(report_target_names)*0.9), max(6, len(report_target_names)*0.8)))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.show()
    except Exception as plot_err:
        print(f"Could not plot confusion matrix for {model_name}: {plot_err}")
        
    return {'model': model_name, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
