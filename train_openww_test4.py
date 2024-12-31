import tensorflow as tf
import librosa
import numpy as np

# Load and Preprocess Audio
audio, sr = librosa.load("test_audio.wav", sr=16000)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
mel_spec = mel_spec[:, :100]  # Ensure consistent size
mel_spec = np.expand_dims(mel_spec, axis=(0, -1))  # Add batch and channel dimensions

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="wakeword_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run Inference
interpreter.set_tensor(input_details[0]['index'], mel_spec.astype(np.float32))
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print("Wake Word Detection Score:", output)
