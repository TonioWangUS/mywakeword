import torch
from train_openww_onnx2 import WakeWordModel

# Load trained model
model = WakeWordModel()
torch.save(model.state_dict(), 'wakeword_model.pth')

# Export to ONNX
dummy_input = torch.randn(1, 1, 64, 100)
torch.onnx.export(
    model, 
    dummy_input, 
    "wakeword_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("Model exported as wakeword_model.onnx")
