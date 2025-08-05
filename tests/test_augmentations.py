import pytest
import numpy as np
from src.data.augmentations import add_noise

@pytest.fixture
def sample_audio_signal():
    """Provides a sample audio signal for testing."""
    return np.zeros(22050, dtype=np.float32)

def test_add_noise_shape(sample_audio_signal):
    """Tests that the add_noise augmentation does not change the signal shape."""
    augmented_signal = add_noise(sample_audio_signal, noise_factor=0.005)
    assert augmented_signal.shape == sample_audio_signal.shape

def test_add_noise_changes_signal(sample_audio_signal):
    """Tests that the add_noise augmentation actually changes the signal."""
    augmented_signal = add_noise(sample_audio_signal, noise_factor=0.005)
    # The augmented signal should not be identical to the original silent signal
    assert not np.array_equal(augmented_signal, sample_audio_signal)

def test_add_noise_zero_factor(sample_audio_signal):
    """Tests that a noise factor of zero does not change the signal."""
    augmented_signal = add_noise(sample_audio_signal, noise_factor=0.0)
    assert np.array_equal(augmented_signal, sample_audio_signal)

def test_add_noise_returns_float32(sample_audio_signal):
    """Tests that the augmented signal is of type float32."""
    augmented_signal = add_noise(sample_audio_signal, noise_factor=0.005)
    assert augmented_signal.dtype == np.float32
