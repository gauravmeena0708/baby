import os
import tensorflow as tf
import itertools
from src.utils.config_loader import load_config
from src.data.data_loader import create_dataset
from src.models.cnn import build_cnn_model
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate

def run(config_path="configs/config.yaml"):
    """
    Main function to run the hyperparameter search pipeline.
    """
    # Load configuration
    config = load_config(config_path)

    # Define hyperparameter search space
    search_space = {
        'learning_rate': [0.001, 0.01],
        'batch_size': [2, 4]
    }

    best_accuracy = -1.0
    best_hyperparameters = None

    # Get all combinations of hyperparameters
    keys, values = zip(*search_space.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"--- Starting Hyperparameter Search for {len(hyperparameter_combinations)} combinations ---")

    for i, params in enumerate(hyperparameter_combinations):
        print(f"\n--- Combination {i+1}/{len(hyperparameter_combinations)}: {params} ---")

        # Update config with new hyperparameters
        config["model"]["params"]["learning_rate"] = params["learning_rate"]
        config["training"]["batch_size"] = params["batch_size"]

        # Set random seeds for reproducibility
        tf.random.set_seed(config["project"]["seed"])

        # Create datasets
        full_dataset = create_dataset(
            data_dir=config["data"]["data_path"],
            class_names=config["data"]["class_names"],
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            n_mels=config["data"]["audio"]["n_mels"]
        )

        dataset_size = len(list(full_dataset))
        val_size = int(config["data"]["validation_split"] * dataset_size)
        train_dataset = full_dataset.take(dataset_size - val_size)
        val_dataset = full_dataset.skip(dataset_size - val_size)

        # Build and compile the model
        for features, _ in train_dataset.take(1):
            input_shape = features.shape[1:]
            break

        model = build_cnn_model(
            input_shape=input_shape,
            num_classes=config["model"]["params"]["num_classes"]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Train the model (a simplified loop for this example)
        for epoch in range(config["training"]["epochs"]):
            print(f"Epoch {epoch+1}/{config['training']['epochs']}")
            train_one_epoch(model, train_dataset, loss_fn, optimizer)

        # Evaluate the model
        loss, accuracy = evaluate(model, val_dataset, loss_fn)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Update best hyperparameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameters = params

    print("\n--- Hyperparameter Search Finished ---")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Hyperparameters: {best_hyperparameters}")

if __name__ == "__main__":
    run()
