from openwakeword import Model

# Load the model
model_path = "./wakeword_model.tflite"
detector = Model(model_path)

# Test audio file
result = detector.predict("test_audio.wav")

if result['wakeword_detected']:
    print("Wake word detected!")
else:
    print("No wake word detected.")
