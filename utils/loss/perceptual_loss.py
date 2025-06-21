import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.resize = resize

        # Use pretrained VGG19 model with new weights parameter
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # We'll only use features up to relu5_4 (layer 35)
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:36]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # Normalization used during VGG training
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr (Tensor): Super-resolved image (B, 3, H, W), range [0, 1]
            hr (Tensor): Ground truth high-res image (B, 3, H, W), range [0, 1]

        Returns:
            perceptual_loss (Tensor): scalar loss value
        """
        # Resize if needed (VGG expects 224x224 or larger)
        if self.resize:
            sr = nn.functional.interpolate(sr, size=(224, 224), mode='bilinear', align_corners=False)
            hr = nn.functional.interpolate(hr, size=(224, 224), mode='bilinear', align_corners=False)

        sr = self.transform(sr)
        hr = self.transform(hr)

        # Extract VGG features
        sr_features = self.vgg_layers(sr)
        hr_features = self.vgg_layers(hr)

        # Compute L1 loss in feature space
        loss = nn.functional.l1_loss(sr_features, hr_features)
        return loss
