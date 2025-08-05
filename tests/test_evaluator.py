import pytest
import tensorflow as tf
from src.models.cnn import build_cnn_model
from src.engine.evaluator import evaluate

@pytest.fixture
def dummy_dataset():
    """Provides a dummy dataset for evaluation."""
    def dummy_generator():
        for _ in range(10):
            yield tf.random.uniform((2, 128, 44, 1)), tf.one_hot(tf.random.uniform((2,), maxval=9, dtype=tf.int32), 9)

    return tf.data.Dataset.from_generator(
        dummy_generator,
        output_signature=(
            tf.TensorSpec(shape=(2, 128, 44, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(2, 9), dtype=tf.float32)
        )
    )

@pytest.fixture
def dummy_model():
    """Provides a dummy CNN model."""
    return build_cnn_model(input_shape=(128, 44, 1), num_classes=9)

def test_evaluate_returns_metrics(dummy_model, dummy_dataset):
    """Tests that the evaluate function returns loss and accuracy."""
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    loss, accuracy = evaluate(dummy_model, dummy_dataset, loss_fn)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
