import pytest
import tensorflow as tf
import numpy as np
import os
from src.models.cnn import build_cnn_model
from src.data.data_loader import create_dataset
from src.engine.trainer import train_one_epoch

@pytest.fixture
def dummy_dataset():
    """Provides a dummy dataset for training."""
    # It's easier to create a dummy dataset from random tensors
    # than to use the full data loader for this test.
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

def test_weights_are_updated(dummy_model, dummy_dataset):
    """Tests that model weights are updated after a training epoch."""
    initial_weights = [tf.identity(w) for w in dummy_model.trainable_variables]

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    train_one_epoch(dummy_model, dummy_dataset, loss_fn, optimizer)

    updated_weights = dummy_model.trainable_variables

    # Check that at least one weight has changed
    for initial, updated in zip(initial_weights, updated_weights):
        if not tf.reduce_all(tf.equal(initial, updated)):
            assert True
            return

    pytest.fail("Model weights were not updated after training.")

def test_checkpointing(dummy_model, dummy_dataset):
    """Tests that model checkpoints can be saved and loaded."""
    checkpoint_dir = "tests/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Train for one epoch
    train_one_epoch(dummy_model, dummy_dataset, loss_fn, optimizer)

    # Save a checkpoint
    checkpoint = tf.train.Checkpoint(model=dummy_model)
    checkpoint.save(os.path.join(checkpoint_dir, "ckpt"))

    # Create a new model
    new_model = build_cnn_model(input_shape=(128, 44, 1), num_classes=9)

    # Load the checkpoint
    load_checkpoint = tf.train.Checkpoint(model=new_model)
    load_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Compare the weights
    for original_w, loaded_w in zip(dummy_model.trainable_variables, new_model.trainable_variables):
        assert tf.reduce_all(tf.equal(original_w, loaded_w))

    # Clean up the checkpoint directory
    tf.io.gfile.rmtree(checkpoint_dir)
