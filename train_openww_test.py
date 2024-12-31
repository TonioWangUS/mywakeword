from openwakeword.model import WakeWordModel

model_path = "./wakeword_model.tflite"
detector = WakeWordModel(model_path)

# Test Detection
result = detector.predict("test_audio.wav")
if result["wakeword_detected"]:
    print("Wake word detected!")
else:
    print("No wake word detected.")
