# gradcam.py
import torch
import cv2
import numpy as np
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[0, target_class_idx].backward(retain_graph=True)
        # self.gradients and self.feature_maps are now set
        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError("Failed to capture gradients or feature maps")
        
        weights = torch.mean(self.gradients[0], dim=(1, 2))  # [C]
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i, :, :]
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # Resize to input size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        return cam

def grad_cam(model, input_tensor, target_class_idx, device):
    """Wrapper for GradCAM using the last conv layer of DenseNet"""
    # Target layer: DenseNet's last convolutional block (features.norm5)
    target_layer = model.backbone.features.norm5
    gradcam = GradCAM(model, target_layer)
    return gradcam(input_tensor, target_class_idx)

def overlay_heatmap(image_pil, heatmap, alpha=0.5):
    img = np.array(image_pil.convert('RGB'))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlayed)

def encode_to_base64(pil_image):
    import io, base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()