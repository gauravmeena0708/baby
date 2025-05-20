import numpy as np
import librosa
from . import config
from tqdm import tqdm
import pandas as pd
import warnings
# --- Feature Set from Cell 2/3/6/7 (50 MFCCs, mean aggregated) ---
def extract_mfcc_means_cell2(audio, sample_rate):
    if audio is None or len(audio) < config.N_FFT: return None
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=config.N_MFCC_CELL2, 
                                          n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return np.mean(mfccs_features.T, axis=0)

# --- Feature Set from Cell 4 (20 MFCCs + ZCR + RMS, mean aggregated) ---
def extract_features_cell4(audio, sample_rate):
    if audio is None or len(audio) < config.N_FFT: return None
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=config.N_MFCC_CELL4,
                                 n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH).T, axis=0)
    
    return np.concatenate((mfccs_mean, zcr, rms))

# --- Feature Set from Cell 5 (Stats of MFCC, ZCR, RMS) ---
def _compute_stats(feature_sequence):
    if feature_sequence.size == 0:
         return np.zeros(feature_sequence.shape[1]*2 if feature_sequence.ndim > 1 else 2)
    return np.hstack((np.mean(feature_sequence, axis=0), np.std(feature_sequence, axis=0)))

def extract_feature_stats_cell5(audio, sample_rate):
    if audio is None or len(audio) < config.N_FFT: return None
    # MFCC sequence
    mfccs_s = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=config.N_MFCC_CELL5, 
                                   n_fft=config.N_FFT, hop_length=config.HOP_LENGTH).T
    # ZCR sequence
    zcr_s = librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH).T
    # RMS sequence
    rms_s = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH).T

    mfcc_stats = _compute_stats(mfccs_s)
    zcr_stats = _compute_stats(zcr_s)
    rms_stats = _compute_stats(rms_s)
    
    return np.hstack((mfcc_stats, zcr_stats, rms_stats))


# --- Feature Set from Cell 6 (Re-implementation of Paper's features) ---
# This combines MFCC, Mel Spec, Chroma, Spectral Contrast, ZCR, RMS means
def extract_combined_paper_like_features(audio, sample_rate):
    if audio is None or len(audio) < config.N_FFT: return None
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, 
                                 n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, 
                                              n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    mel_spec_mean = np.mean(mel_spec.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, 
                                         n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    chroma_mean = np.mean(chroma.T, axis=0)

    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, 
                                                      n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    spec_contrast_mean = np.mean(spec_contrast.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH)
    zcr_mean = np.mean(zcr.T, axis=0)

    rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)
    rms_mean = np.mean(rms.T, axis=0)

    all_features = np.hstack((
        mfccs_mean, mfccs_std, mel_spec_mean, chroma_mean,
        spec_contrast_mean, zcr_mean, rms_mean
    ))
    return all_features

# --- Augmentation ---
def add_gaussian_noise(audio_segment, noise_factor=config.NOISE_FACTOR):
    if audio_segment is None: return None
    noise = np.random.randn(len(audio_segment))
    augmented_audio = audio_segment + noise_factor * noise
    augmented_audio = np.clip(augmented_audio, -1., 1.) # Clip to valid audio range
    return augmented_audio

def process_features_for_all_audio(audio_segments, labels, feature_type="mfcc_cell2", augment=False):
    """
    Extracts specified features for all audio segments.
    Optionally applies noise augmentation to classes not in CLASSES_TO_AUGMENT_LESS.
    """
    print(f"\nExtracting features of type: {feature_type} (Augment: {augment})...")
    extracted_data = []
    
    for i, audio in enumerate(tqdm(audio_segments, desc="Feature Extraction")):
        original_label_idx = labels[i] # Original label index
        
        # Original features
        if feature_type == "mfcc_cell2":
            feats = extract_mfcc_means_cell2(audio, config.SAMPLE_RATE)
        elif feature_type == "features_cell4":
            feats = extract_features_cell4(audio, config.SAMPLE_RATE)
        elif feature_type == "stats_cell5":
            feats = extract_feature_stats_cell5(audio, config.SAMPLE_RATE)
        elif feature_type == "combined_paper_like":
            feats = extract_combined_paper_like_features(audio, config.SAMPLE_RATE)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        if feats is not None:
            extracted_data.append([feats, original_label_idx])

        # Augmentation (applied to the original audio segment)
        if augment:
            original_label_name = config.CLASS_LABELS_LIST[original_label_idx]
            if original_label_name not in config.CLASSES_TO_AUGMENT_LESS:
                noisy_audio = add_gaussian_noise(audio, noise_factor=config.NOISE_FACTOR)
                if noisy_audio is not None:
                    if feature_type == "mfcc_cell2":
                        feats_aug = extract_mfcc_means_cell2(noisy_audio, config.SAMPLE_RATE)
                    elif feature_type == "features_cell4":
                        feats_aug = extract_features_cell4(noisy_audio, config.SAMPLE_RATE)
                    elif feature_type == "stats_cell5":
                        feats_aug = extract_feature_stats_cell5(noisy_audio, config.SAMPLE_RATE)
                    elif feature_type == "combined_paper_like":
                        feats_aug = extract_combined_paper_like_features(noisy_audio, config.SAMPLE_RATE)
                    
                    if feats_aug is not None:
                        extracted_data.append([feats_aug, original_label_idx])
                        
    if not extracted_data:
        raise ValueError("No features were extracted after processing. Check audio files and feature extraction logic.")
        
    features_df = pd.DataFrame(extracted_data, columns=['feature', 'class'])
    print(f"\nSuccessfully extracted features for {len(features_df)} segments (including augmentations if any).")
    if not features_df.empty:
        print(f"Shape of feature vector for first sample: {features_df['feature'].iloc[0].shape}")
    return features_df
