import os
import pathlib
import librosa
import numpy as np
from tqdm import tqdm
from . import config

def get_audio_files(base_dir, class_indexes):
    """
    Lists all audio files and their corresponding class indices.
    """
    audio_files_info = []
    sub_dirs_info = [(class_name, os.path.join(base_dir, class_name)) for class_name in class_indexes.keys()]

    for class_name, dir_path in sub_dirs_info:
        print(f"Checking directory: {dir_path}")
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory not found - {dir_path}")
            continue
        try:
            p = pathlib.Path(dir_path)
            files_in_dir = list(p.iterdir())
            if not files_in_dir:
                print(f"Warning: No files found in {dir_path}")
            for file_path_obj in files_in_dir:
                if file_path_obj.is_file() and file_path_obj.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                    audio_files_info.append((class_indexes[class_name], file_path_obj))
                # else:
                #     print(f"Skipping non-audio or non-file item: {file_path_obj}")
        except Exception as e:
            print(f"Error iterating directory {dir_path}: {e}")
    
    print(f"\nFound {len(audio_files_info)} audio files to process.")
    return audio_files_info

def load_audio_segment(file_path, sr=config.SAMPLE_RATE, duration=config.DURATION_SECONDS, fixed_duration=config.FIXED_DURATION_PROCESSING):
    """
    Loads an audio file, optionally resamples, and pads/truncates to a fixed duration.
    """
    try:
        if fixed_duration and duration:
            audio, current_sr = librosa.load(file_path, sr=sr, duration=duration)
            max_len = int(duration * sr)
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
            elif len(audio) > max_len:
                audio = audio[:max_len]
        else:
            audio, current_sr = librosa.load(file_path, sr=sr)

        if len(audio) < config.N_FFT : # Ensure enough samples for FFT based processing
            # print(f"Warning: Skipping very short file after loading: {file_path}, length {len(audio)}")
            return None, None
            
        return audio, current_sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def load_all_audio_data(audio_files_info):
    """
    Loads all audio segments and their labels from the provided file info.
    """
    print("Loading and processing audio segments...")
    all_audio_segments = []
    all_labels = []
    
    skipped_files = 0
    for class_idx, file_path_obj in tqdm(audio_files_info, desc="Loading Audio"):
        audio, sr = load_audio_segment(file_path_obj)
        if audio is not None:
            all_audio_segments.append(audio)
            all_labels.append(class_idx)
        else:
            skipped_files += 1
            
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files due to loading or length issues.")
        
    return all_audio_segments, np.array(all_labels)
