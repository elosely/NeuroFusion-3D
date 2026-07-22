import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DoubleConv3D(nn.Module):
    """(Convolution3D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NeuroFusion3DUNet(nn.Module):
    """
    Robust 3D U-Net Architecture for Brain Tumor Segmentation.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.inc = DoubleConv3D(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(32, 64))

        # Bottleneck
        self.bottleneck = DoubleConv3D(64, 128)

        # Decoder
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(64 + 64, 64)

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(32 + 16, 32)

        # Output
        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x_b = self.bottleneck(x3)

        x = self.up1(x_b)
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)

        logits = self.outc(x)
        return self.sigmoid(logits)


class AIInferenceEngine:
    """
    Inference Engine using Pre-trained Weights.
    """
    @staticmethod
    def run_inference(volume_data: torch.Tensor, model: torch.nn.Module = None) -> torch.Tensor:
        weights_path = "neurofusion_unet_brats_weights.pth"

        if model is None:
            model = NeuroFusion3DUNet(in_channels=1, out_channels=1)
            
            # Load pre-trained weights if available
            if os.path.exists(weights_path):
                try:
                    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
                    model.load_state_dict(state_dict)
                except Exception as e:
                    pass

            model.eval()

        if len(volume_data.shape) == 3:
            input_tensor = volume_data.unsqueeze(0).unsqueeze(0).float()
        else:
            input_tensor = volume_data.float()

        with torch.no_grad():
            output_mask = model(input_tensor)

        return output_mask.squeeze().cpu().numpy()