# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ChestXRayClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        # For Grad-CAM: store feature maps of last conv layer
        self.feature_maps = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.feature_maps = output
        self.backbone.features.norm5.register_forward_hook(hook_fn)

    def forward(self, x):
        logits = self.backbone(x)
        return logits

    def get_probs(self, x):
        return torch.sigmoid(self.forward(x))