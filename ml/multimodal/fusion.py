import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class NeuralMultimodalFusionNetwork(nn.Module):
    def __init__(self, tab_dim=15, img_dim=512, text_dim=64, fused_dim=64):
        super().__init__()
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(96, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, tab_feat, img_feat, text_feat):
        t_emb = self.tab_encoder(tab_feat)
        i_emb = self.img_encoder(img_feat)
        x_emb = self.text_encoder(text_feat)
        
        fused_feat = torch.cat([t_emb, i_emb, x_emb], dim=1)
        output = self.fusion_head(fused_feat)
        return output

class MultimodalFusionEngine:
    def __init__(self, tabular_weight=0.35, vision_weight=0.40, nlp_weight=0.25):
        self.w_tab = tabular_weight
        self.w_img = vision_weight
        self.w_nlp = nlp_weight
        
        self.neural_fusion = NeuralMultimodalFusionNetwork()
        self.neural_fusion.eval()

    def predict_late_fusion(self, tabular_risk: float, image_risk: float, nlp_risk: float) -> float:
        total_w = self.w_tab + self.w_img + self.w_nlp
        fused = (self.w_tab * tabular_risk + self.w_img * image_risk + self.w_nlp * nlp_risk) / total_w
        return round(float(fused), 4)

    def predict_neural_fusion(self, tab_embed: np.ndarray, img_embed: np.ndarray, text_embed: np.ndarray) -> float:
        with torch.no_grad():
            t_tensor = torch.tensor(tab_embed, dtype=torch.float32).unsqueeze(0) if tab_embed.ndim == 1 else torch.tensor(tab_embed, dtype=torch.float32)
            i_tensor = torch.tensor(img_embed, dtype=torch.float32).unsqueeze(0) if img_embed.ndim == 1 else torch.tensor(img_embed, dtype=torch.float32)
            x_tensor = torch.tensor(text_embed, dtype=torch.float32).unsqueeze(0) if text_embed.ndim == 1 else torch.tensor(text_embed, dtype=torch.float32)
            
            fused_score = self.neural_fusion(t_tensor, i_tensor, x_tensor).item()
            return round(float(fused_score), 4)

    def fuse_all_modalities(
        self,
        tabular_risk: float,
        image_risk: float,
        nlp_risk: float,
        tab_embed: np.ndarray = None,
        img_embed: np.ndarray = None,
        text_embed: np.ndarray = None,
        method="neural"
    ) -> Dict[str, Any]:
        late_fused = self.predict_late_fusion(tabular_risk, image_risk, nlp_risk)
        
        if method == "neural" and tab_embed is not None and img_embed is not None and text_embed is not None:
            neural_fused = self.predict_neural_fusion(tab_embed, img_embed, text_embed)
            final_fused = neural_fused
        else:
            final_fused = late_fused

        return {
            'tabular_risk': tabular_risk,
            'image_risk': image_risk,
            'nlp_risk': nlp_risk,
            'late_fused_risk': late_fused,
            'final_fused_risk': final_fused,
            'fusion_strategy': method,
            'disclaimer': "Research decision-support prediction only. Not a clinical diagnosis."
        }
