import pytest
import numpy as np
from src.data.feature_extractor import load_audio, to_mel_spectrogram

@pytest.fixture
def audio_file_path():
    """Provides the path to a dummy audio file."""
    return "tests/dummy_data/hungry/silent_0.wav"

def test_load_audio(audio_file_path):
    """Tests that an audio file can be loaded and has the correct shape and type."""
    signal, sample_rate = load_audio(audio_file_path, sample_rate=22050)
    assert isinstance(signal, np.ndarray)
    assert signal.dtype == np.float32
    assert sample_rate == 22050
    assert len(signal) > 0

def test_to_mel_spectrogram(audio_file_path):
    """Tests that a Mel spectrogram can be created with the correct shape."""
    signal, sr = load_audio(audio_file_path, sample_rate=22050)

    # Example parameters, these should be configurable in a real scenario
    n_mels = 128
    n_fft = 2048
    hop_length = 512

    mel_spec = to_mel_spectrogram(signal, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    assert isinstance(mel_spec, np.ndarray)
    assert mel_spec.dtype == np.float32

    # Calculate expected time steps
    expected_time_steps = int(np.floor(len(signal) / hop_length)) + 1
    assert mel_spec.shape == (n_mels, expected_time_steps)

def test_load_audio_file_not_found():
    """Tests that an error is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_audio("non_existent_file.wav", 22050)
