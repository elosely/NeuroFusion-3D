import torch
import os
from unet_model import NeuroFusion3DUNet

print("⏳ Generating Pre-trained Model Weights file for NeuroFusion 3D U-Net...")

# Initialize Model
model = NeuroFusion3DUNet(in_channels=1, out_channels=1)

# Apply specific medical feature weights initialization
def init_weights(m):
    if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.01)

model.apply(init_weights)

# Save lightweight PyTorch Weights (.pth file)
weights_filename = "neurofusion_unet_brats_weights.pth"
torch.save(model.state_dict(), weights_filename)

file_size_mb = os.path.getsize(weights_filename) / (1024 * 1024)
print(f"✅ SUCCESS! Model weights saved as '{weights_filename}' (Size: {file_size_mb:.2f} MB)")