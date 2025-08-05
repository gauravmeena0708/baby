# Baby Cry Classification (Refactored)

This project is a refactored version of a baby cry classification system, built with a strong emphasis on modern Machine Learning Engineering best practices and a strict Test-Driven Development (TDD) methodology. The goal is to provide a robust, maintainable, and reproducible system for classifying baby cry audio.

## Project Structure

The project is organized into a modular structure that separates concerns and promotes reusability:

- `configs/`: Contains configuration files. All experiment parameters are managed via `config.yaml`.
- `data/`: Contains the raw audio data (not checked into Git). A dummy dataset for testing is available in `tests/dummy_data`.
- `outputs/`: Default directory for all artifacts generated during runs, including checkpoints, logs, and saved models.
- `src/`: Contains all the source code, organized into logical modules:
  - `data/`: Data loading, feature extraction, and augmentation.
  - `engine/`: Training and evaluation logic.
  - `models/`: Model definitions.
  - `utils/`: Utility functions, such as the configuration loader.
- `tests/`: Contains all unit and integration tests, following the TDD methodology.
- `run_*.py`: Executable scripts for running different pipelines.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Set up a Python environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project uses a `setup.py` file to manage dependencies. Install the project in editable mode:
    ```bash
    pip install -e .
    ```

4.  **Data:**
    - The project is configured to run with the dummy data in `tests/dummy_data`.
    - To run with the full dataset, download the data and update the `data_path` in `configs/config.yaml` to point to your data directory. The directory should have subdirectories for each class (e.g., "hungry", "pain").

## Running the Pipeline

The main entry points for the pipeline are the `run_*.py` scripts.

### Training

To run the training pipeline, use the `run_training.py` script. All training parameters can be configured in `configs/config.yaml`.

```bash
python run_training.py
```
This will train the model, save checkpoints, and save the final trained model to the `outputs` directory.

### Evaluation

To evaluate a trained model, use the `run_evaluation.py` script. This will load the final model from the `outputs/models` directory and evaluate it on the dataset specified in the configuration.

```bash
python run_evaluation.py
```

### Hyperparameter Search

To run a hyperparameter search, use the `run_hyperparameter_search.py` script. The search space is defined within the script itself.

```bash
python run_hyperparameter_search.py
```

## Test-Driven Development (TDD)

This project was refactored using a strict TDD workflow. For each component (e.g., data loader, model, trainer), a comprehensive suite of tests was written *before* the implementation. This ensures that every part of the system is robust, verifiable, and functions as expected.

To run all tests, use `pytest`:
```bash
python -m pytest
```
