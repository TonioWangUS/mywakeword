import os
import librosa
import numpy as np

DATASET_PATH = "./dataset"
OUTPUT_PATH = "./spectrograms"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Fixed spectrogram time length (e.g., 100 time frames)
FIXED_TIME_FRAMES = 100

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Adjust time axis to FIXED_TIME_FRAMES
    if log_mel_spec.shape[1] < FIXED_TIME_FRAMES:
        # Pad with zeros if shorter
        pad_width = FIXED_TIME_FRAMES - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Trim if longer
        log_mel_spec = log_mel_spec[:, :FIXED_TIME_FRAMES]
    
    return log_mel_spec

# Process Dataset
data = []
labels = []

for category, label in [('wake_word', 1), ('background', 0), ('silence', 0)]:
    folder = os.path.join(DATASET_PATH, category)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        spectrogram = process_audio(file_path)
        data.append(spectrogram)
        labels.append(label)

# Save preprocessed data
np.save(os.path.join(OUTPUT_PATH, 'data.npy'), np.array(data))
np.save(os.path.join(OUTPUT_PATH, 'labels.npy'), np.array(labels))
