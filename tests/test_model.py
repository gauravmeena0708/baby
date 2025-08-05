import pytest
import tensorflow as tf
from src.models.cnn import build_cnn_model

@pytest.fixture
def sample_input():
    """Provides a sample input tensor for the model."""
    # (batch_size, n_mels, time_steps, channels)
    # The data loader produces (batch, n_mels, time_steps), so we need to add a channel dimension.
    return tf.random.uniform((2, 128, 44, 1), dtype=tf.float32)

@pytest.fixture
def model_config():
    """Provides a sample model configuration."""
    return {
        "input_shape": (128, 44, 1),
        "num_classes": 9
    }

def test_model_output_shape(sample_input, model_config):
    """Tests that the model's output has the correct shape."""
    model = build_cnn_model(
        input_shape=model_config["input_shape"],
        num_classes=model_config["num_classes"]
    )
    output = model(sample_input)

    expected_shape = (sample_input.shape[0], model_config["num_classes"])
    assert output.shape == expected_shape

def test_model_output_dtype(sample_input, model_config):
    """Tests that the model's output has the correct data type."""
    model = build_cnn_model(
        input_shape=model_config["input_shape"],
        num_classes=model_config["num_classes"]
    )
    output = model(sample_input)
    assert output.dtype == tf.float32

def test_model_forward_pass(sample_input, model_config):
    """Ensures the model's forward pass executes without errors."""
    try:
        model = build_cnn_model(
            input_shape=model_config["input_shape"],
            num_classes=model_config["num_classes"]
        )
        model(sample_input)
    except Exception as e:
        pytest.fail(f"Model forward pass failed with exception: {e}")
