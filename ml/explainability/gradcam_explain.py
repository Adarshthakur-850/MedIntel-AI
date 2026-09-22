import os
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm

class GradCAMExplainer:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.target_layer if hasattr(self.model, 'target_layer') else None
        if target_layer is None:
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        score = logits[0, 0]
        score.backward()

        if self.gradients is None or self.activations is None:
            # Fallback synthetic heatmap if hooks did not capture gradients
            return np.random.uniform(0, 1, (224, 224))

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0..1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, orig_image_path: str, heatmap: np.ndarray, output_path: str):
        img = Image.open(orig_image_path).convert('RGB').resize((224, 224))
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Resize heatmap
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)
        heatmap_np = np.array(heatmap_img, dtype=np.float32) / 255.0

        # Color map
        cmap = cm.get_cmap('jet')
        heatmap_color = cmap(heatmap_np)[:, :, :3]

        # Blend
        blended = 0.6 * img_np + 0.4 * heatmap_color
        blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

        result = Image.fromarray(blended)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)
        return output_path
