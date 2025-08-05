import librosa
import numpy as np
import os

def load_audio(file_path: str, sample_rate: int):
    """
    Loads an audio file.

    Args:
        file_path: The path to the audio file.
        sample_rate: The sample rate to use for loading the audio.

    Returns:
        A tuple of (signal, sample_rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    signal, sr = librosa.load(file_path, sr=sample_rate)
    return signal.astype(np.float32), sr

def to_mel_spectrogram(signal, sample_rate, n_mels, n_fft, hop_length):
    """
    Converts an audio signal to a Mel spectrogram.

    Args:
        signal: The audio signal.
        sample_rate: The sample rate of the audio signal.
        n_mels: The number of Mel bands.
        n_fft: The FFT window size.
        hop_length: The hop length.

    Returns:
        The Mel spectrogram.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mel_spectrogram.astype(np.float32)
