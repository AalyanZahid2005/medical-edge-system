import torch
import torch.nn as nn

class MedicalOODAutoencoder(nn.Module):
    def __init__(self):
        super(MedicalOODAutoencoder, self).__init__()
        
        # ---------------------------------------------------------
        # ENCODER: Compresses 224x224x3 down to 14x14x128
        # ---------------------------------------------------------
        self.encoder = nn.Sequential(
            # Input: [Batch, 3, 224, 224]
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: [16, 112, 112]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: [32, 56, 56]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: [64, 28, 28]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: [128, 14, 14]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ---------------------------------------------------------
        # DECODER: Rebuilds 14x14x128 back to 224x224x3
        # ---------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # [64, 28, 28]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [32, 56, 56]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [16, 112, 112]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # [3, 224, 224]
            nn.Sigmoid() # Squashes output pixel values between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# --- Quick Test Block ---
# This ensures the tensor math and dimensions line up perfectly.
if __name__ == "__main__":
    print("[SYSTEM] Testing OOD Autoencoder Architecture...")
    model = MedicalOODAutoencoder()
    
    # Simulate a dummy batch of 4 images, 3 channels, 224x224
    dummy_input = torch.randn(4, 3, 224, 224) 
    
    output = model(dummy_input)
    
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    if dummy_input.shape == output.shape:
        print("[SUCCESS] Architecture verified. Encoder and Decoder shapes match.")
    else:
        print("[FATAL ERROR] Dimension mismatch.")