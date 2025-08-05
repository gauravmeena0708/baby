import os
import tensorflow as tf
from src.utils.config_loader import load_config
from src.data.data_loader import create_dataset
from src.models.cnn import build_cnn_model

def run(config_path="configs/config.yaml"):
    """
    Main function to run the training pipeline.
    """
    # Load configuration
    config = load_config(config_path)

    # Set random seeds for reproducibility
    tf.random.set_seed(config["project"]["seed"])

    # Create datasets
    # For simplicity, this example uses the same directory for train and val.
    # A more robust pipeline would have separate train/val/test splits.
    full_dataset = create_dataset(
        data_dir=config["data"]["data_path"],
        class_names=config["data"]["class_names"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        n_mels=config["data"]["audio"]["n_mels"],
        n_fft=config["data"]["audio"]["n_fft"],
        hop_length=config["data"]["audio"]["hop_length"],
        sample_rate=config["data"]["audio"]["sample_rate"]
    )

    # Splitting the dataset
    # This is a basic way to split a tf.data.Dataset.
    # A more robust approach would be to split the file paths beforehand.
    dataset_size = len(list(full_dataset))
    val_size = int(config["data"]["validation_split"] * dataset_size)
    train_size = dataset_size - val_size

    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)

    # Build the model
    # Infer input shape from the dataset
    for features, _ in train_dataset.take(1):
        input_shape = features.shape[1:]
        break

    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=config["model"]["params"]["num_classes"]
    )

    model.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["model"]["params"]["learning_rate"])
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Create callbacks
    # Create output directories if they don't exist
    output_dir = config["outputs"]["output_dir"]
    checkpoint_dir = os.path.join(output_dir, config["outputs"]["checkpoint_dir"])
    log_dir = os.path.join(output_dir, config["outputs"]["log_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras"),
        monitor=config["training"]["checkpoint"]["monitor"],
        save_best_only=config["training"]["checkpoint"]["save_best_only"],
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=config["training"]["early_stopping"]["monitor"],
        patience=config["training"]["early_stopping"]["patience"],
        restore_best_weights=True,
        verbose=1
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Train the model
    print("\n--- Starting Training ---")
    model.fit(
        train_dataset,
        epochs=config["training"]["epochs"],
        validation_data=val_dataset,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback]
    )
    print("--- Training Finished ---\n")

    # Save the final model
    saved_model_path = os.path.join(output_dir, config["outputs"]["saved_model_dir"], "final_model.keras")
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    model.save(saved_model_path)
    print(f"Final model saved to: {saved_model_path}")


if __name__ == "__main__":
    run()
