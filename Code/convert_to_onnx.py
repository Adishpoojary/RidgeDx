"""Convert PyTorch model to ONNX for lightweight deployment."""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import torch.nn as nn
from torchvision import models

# Build model
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 8)

# Load trained weights
model.load_state_dict(torch.load('resnet18_bloodgroup.pth', map_location='cpu', weights_only=True))
model.eval()

# Export to ONNX using legacy exporter (more reliable)
dummy_input = torch.randn(1, 1, 128, 128)
torch.onnx.export(
    model, dummy_input, 'resnet18_bloodgroup.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=13,
    dynamo=False
)

size_mb = os.path.getsize('resnet18_bloodgroup.onnx') / 1e6
print(f'ONNX model saved: {size_mb:.1f} MB')

# Verify
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession('resnet18_bloodgroup.onnx')
test_input = np.random.randn(1, 1, 128, 128).astype(np.float32)
result = session.run(None, {'input': test_input})
print(f'Output shape: {result[0].shape}')
print(f'Predictions: {result[0][0][:4]}...')
print('Verification passed!')
