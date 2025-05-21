# Baby Cry Classification

This project aims to classify baby cry sounds into different categories (e.g., hungry, pain, discomfort) using audio features and machine learning/deep learning models.

## Project Structure

- `data/`: (Optional) For storing raw audio data if not using Kaggle/mounted paths.
- `notebooks/`: Contains the original Jupyter Notebook for reference.
- `src/`: Contains the Python source code.
  - `config.py`: Configuration parameters (paths, model hyperparameters, etc.).
  - `data_loader.py`: Scripts for loading and initial processing of audio data.
  - `feature_extractor.py`: Scripts for feature extraction and augmentation.
  - `models.py`: Definitions for Keras (Deep Learning) and Scikit-learn (Machine Learning) models.
  - `train_utils.py`: Utility functions for training models, including data splitting, scaling, and SMOTE.
  - `evaluate_utils.py`: Scripts for evaluating model performance and generating plots.
  - `main_dl_smote.py`: Example main script to run the Deep Learning (Conv1D-LSTM) experiment with SMOTE.
  - `main_ml_augmented.py`: Example main script to run a Machine Learning (Random Forest) experiment with augmented data.
- `requirements.txt`: Lists the Python dependencies for this project.
- `README.md`: This file.

## Setup

1.  **Clone the repository (if applicable).**
2.  **Set up a Python environment:**
    It's recommended to use a virtual environment (e.g., venv, conda).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data:**
    -   Ensure your audio data is accessible. The `src/config.py` file attempts to use a Kaggle input path first (`/kaggle/input/baby-crying-sounds-datasets/Baby Crying Sounds/`).
    -   If running locally, modify `LOCAL_DATA_PATH` in `src/config.py` to point to your "Baby Crying Sounds" directory, or place the data in `./data/Baby Crying Sounds/` relative to the project root. The directory should have subdirectories for each class (e.g., "belly pain", "hungry").

## Running Experiments

You can run specific experiments by executing the corresponding `main_*.py` scripts from the project root directory.

**Example 1: Run Deep Learning experiment with SMOTE (Conv1D-LSTM with 50 MFCC features):**
```bash
python -m src.main_dl_smote
```
```bash
python -m src.main_uber_cry_classifier
```
