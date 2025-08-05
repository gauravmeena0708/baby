import tensorflow as tf
import os
from .feature_extractor import load_audio, to_mel_spectrogram

def create_dataset(data_dir, class_names, batch_size, shuffle=False, shuffle_buffer=1000, n_mels=128, n_fft=2048, hop_length=512, sample_rate=22050):
    """
    Creates a tf.data.Dataset for the audio data.

    Args:
        data_dir: The directory containing the class-named subdirectories of audio files.
        class_names: A list of the class names.
        batch_size: The batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer: The buffer size for shuffling.
        n_mels: The number of Mel bands for the spectrogram.
        n_fft: The FFT window size.
        hop_length: The hop length for the spectrogram.
        sample_rate: The sample rate for loading audio.

    Returns:
        A tf.data.Dataset object.
    """

    # Create a dataset of file paths
    file_paths = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".wav"):
                file_paths.append(os.path.join(class_dir, fname))

    if not file_paths:
        raise ValueError(f"No .wav files found in {data_dir}")

    list_ds = tf.data.Dataset.from_tensor_slices(file_paths)

    def _parse_function(file_path):
        # The file path is a tensor, so we need to use tf.py_function
        # to use our numpy-based feature extraction functions.

        def _load_and_process(path):
            path = path.numpy().decode('utf-8')
            # Get label from path
            parts = path.split(os.sep)
            # The label is the second to last part (e.g., .../hungry/silent_0.wav)
            label_str = parts[-2]
            label = class_names.index(label_str)

            # Load and process audio
            signal, sr = load_audio(path, sample_rate)
            mel_spec = to_mel_spectrogram(signal, sr, n_mels, n_fft, hop_length)

            # TensorFlow expects consistent shapes. We need to pad or truncate spectrograms.
            # For this example, let's assume a fixed length of 1 second of audio.
            # This is a simplification; a real implementation would need a more robust way
            # to handle variable-length audio, such as padding to the max length in a batch.
            expected_time_steps = int(sample_rate / hop_length) + 1

            # Pad or truncate the time dimension
            if mel_spec.shape[1] < expected_time_steps:
                padding = expected_time_steps - mel_spec.shape[1]
                mel_spec = tf.pad(mel_spec, [[0, 0], [0, padding]])
            elif mel_spec.shape[1] > expected_time_steps:
                mel_spec = mel_spec[:, :expected_time_steps]

            return mel_spec, label

        # Wrap the Python function
        mel_spec, label = tf.py_function(
            _load_and_process,
            [file_path],
            [tf.float32, tf.int64]
        )

        # Set the shape of the tensors
        expected_time_steps = int(sample_rate / hop_length) + 1
        mel_spec.set_shape([n_mels, expected_time_steps])
        label.set_shape([])

        # Add a channel dimension for the CNN
        mel_spec = tf.expand_dims(mel_spec, axis=-1)

        # One-hot encode the label
        label_one_hot = tf.one_hot(label, len(class_names))

        return mel_spec, label_one_hot

    # Map the parsing function to the dataset
    dataset = list_ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
