import os
import tensorflow as tf
from src.utils.config_loader import load_config
from src.data.data_loader import create_dataset
from src.engine.evaluator import evaluate

def run(config_path="configs/config.yaml"):
    """
    Main function to run the evaluation pipeline.
    """
    # Load configuration
    config = load_config(config_path)

    # Load the trained model
    model_path = os.path.join(config["outputs"]["output_dir"], config["outputs"]["saved_model_dir"], "final_model.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run the training script first.")
        return

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Create the evaluation dataset
    eval_dataset = create_dataset(
        data_dir=config["data"]["data_path"], # Using the same data for this example
        class_names=config["data"]["class_names"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        n_mels=config["data"]["audio"]["n_mels"],
        n_fft=config["data"]["audio"]["n_fft"],
        hop_length=config["data"]["audio"]["hop_length"],
        sample_rate=config["data"]["audio"]["sample_rate"]
    )

    # Evaluate the model
    print("\n--- Starting Evaluation ---")
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss, accuracy = evaluate(model, eval_dataset, loss_fn)
    print("--- Evaluation Finished ---\n")

    print(f"Evaluation Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run()
