import torch
import torch.nn as nn

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, num_modalities=6, d_model=128, nhead=4):
        super(CrossModalAttentionFusion, self).__init__()
        
        # Project 1280 features from EfficientNet into Transformer space
        self.feature_projections = nn.ModuleList([
            nn.Linear(1280, d_model) for _ in range(num_modalities)
        ])
        
        # Transformer Encoder for cross-modal reasoning 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final Risk Head [cite: 44]
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * num_modalities, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, 6, 1280)
        projected = torch.stack([self.feature_projections[i](x[:, i, :]) for i in range(x.size(1))], dim=1)
        
        # Apply Multi-Head Attention [cite: 213]
        attn_output = self.transformer_encoder(projected)
        
        # Flatten and calculate Risk Score
        attn_output_flat = attn_output.view(attn_output.size(0), -1)
        risk_score = self.risk_head(attn_output_flat)
        
        # Return score AND attention features for the GUI radar chart [cite: 212, 217]
        return risk_score, attn_output

if __name__ == "__main__":
    print("[INFO] Transformer Fusion Architecture Defined.")