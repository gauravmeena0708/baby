import pytest
import tensorflow as tf
import os
from src.data.data_loader import create_dataset

@pytest.fixture
def dummy_data_path():
    """Provides the path to the dummy data directory."""
    return "tests/dummy_data"

def test_create_dataset_file_discovery(dummy_data_path):
    """Tests that the dataset can be created and discovers the correct number of files."""
    # These parameters should match the dummy data created
    class_names = ["hungry", "pain"]

    # This test only checks file discovery, so we can use a small batch size
    dataset = create_dataset(dummy_data_path, class_names, batch_size=1, shuffle=False)

    # Total number of files created was 2 classes * 3 files/class = 6
    num_elements = 0
    for _ in dataset:
        num_elements += 1
    assert num_elements == 6

def test_create_dataset_batch_shape_and_type(dummy_data_path):
    """Tests that the dataset yields batches with the correct shape and data type."""
    class_names = ["hungry", "pain"]
    batch_size = 2
    n_mels = 128 # Example, should be configurable

    dataset = create_dataset(
        dummy_data_path,
        class_names,
        batch_size=batch_size,
        shuffle=False,
        n_mels=n_mels
    )

    # Get one batch from the dataset
    for features, labels in dataset.take(1):
        # Check feature shape: (batch_size, n_mels, time_steps)
        # Time steps can vary, so we only check the first two dimensions
        assert features.shape[0] == batch_size
        assert features.shape[1] == n_mels
        assert features.dtype == tf.float32

        # Check label shape: (batch_size, num_classes)
        assert labels.shape == (batch_size, len(class_names))
        assert labels.dtype == tf.float32

# def test_create_dataset_with_shuffle(dummy_data_path):
#     """Tests that shuffling changes the order of elements."""
#     class_names = ["hungry", "pain"]

#     # Create two datasets, one shuffled and one not
#     unshuffled_dataset = create_dataset(dummy_data_path, class_names, batch_size=6, shuffle=False)
#     shuffled_dataset = create_dataset(dummy_data_path, class_names, batch_size=6, shuffle=True, shuffle_buffer=6)

#     # Get the single batch from each
#     for unshuffled_batch, _ in unshuffled_dataset.take(1):
#         pass
#     for shuffled_batch, _ in shuffled_dataset.take(1):
#         pass

#     # The order of elements should be different
#     # This is not a perfect test for shuffle, but it's a good heuristic
#     assert not tf.reduce_all(tf.equal(unshuffled_batch, shuffled_batch))
