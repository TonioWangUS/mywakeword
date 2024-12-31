from openwakeword import Model

# Correct Path to Your Model
model_path = "./wakeword_model.tflite"

# Initialize WakeWord Detection
detector = Model(wakeword_model_paths=[model_path])

# Test Audio
result = detector.predict("test_audio.wav")
if result['wakeword_detected']:
    print("Wake word detected!")
else:
    print("No wake word detected.")
