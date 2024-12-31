import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load Audio File
audio_file = "wake1.wav"
y, sr = librosa.load(audio_file, sr=16000)

# Generate Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()

