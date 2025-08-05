import numpy as np

def add_noise(signal, noise_factor):
    """
    Adds random noise to an audio signal.

    Args:
        signal: The audio signal as a numpy array.
        noise_factor: The factor by which to scale the noise.

    Returns:
        The noisy signal as a float32 numpy array.
    """
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_factor * noise
    return augmented_signal.astype(np.float32)
