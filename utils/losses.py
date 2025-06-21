import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
import lpips  # pip install lpips


class Losses(nn.Module):
    def __init__(self, 
                 mse_weight=1.0, 
                 lpips_weight=1.0, 
                 vgg_weight=0.1, #  0 for 32x32 images or smaller, 0.1 default for 64x64 or larger images
                 resize_for_vgg=False, # True for 256 or larger images
                 vgg_target_size=224, # standard VGG input size. for 512x512 images, set to 384, for 1024x1024, set to 512
                 device='cuda'):
        super().__init__()

        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.vgg_weight = vgg_weight
        self.resize_for_vgg = resize_for_vgg
        self.vgg_target_size = vgg_target_size

        self.mse = nn.MSELoss()
        
        # Initialize LPIPS once
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

        # Configurable VGG layers
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:36]).to(device)
        self.vgg_layers.eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def vgg_perceptual_loss(self, pred_img, target_img):
        if self.resize_for_vgg:
            pred_img = F.interpolate(
                pred_img, 
                size=(self.vgg_target_size, self.vgg_target_size), 
                mode='bilinear', 
                align_corners=False
            )
            target_img = F.interpolate(
                target_img, 
                size=(self.vgg_target_size, self.vgg_target_size), 
                mode='bilinear', 
                align_corners=False
            )

        # Safe normalization: avoid modifying tensors in-place
        pred_img = self.normalize(pred_img.clone())
        target_img = self.normalize(target_img.clone())

        pred_features = self.vgg_layers(pred_img)
        target_features = self.vgg_layers(target_img)
        return F.l1_loss(pred_features, target_features)

    def forward(self, pred_noise, target_noise, pred_img, target_img):
        loss = 0.0

        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse(pred_noise, target_noise)

        if self.lpips_weight > 0:
            # Detach if using multiple GPUs to avoid backprop through LPIPS
            with torch.no_grad():
                lpips_loss = self.lpips(pred_img, target_img)
            loss += self.lpips_weight * lpips_loss.mean()

        if self.vgg_weight > 0:
            loss += self.vgg_weight * self.vgg_perceptual_loss(pred_img, target_img)

        return loss
    


# resize_for_vgg	vgg_target_size	vgg_weight	lpips_weight	Additional Recommendations
# 28×28	False	-	0.0	0.5-1.0	Skip VGG entirely, use MSE+LPIPS
# 32×32	False	-	0.0-0.1	1.0	LPIPS works better than VGG here
# 64×64	False	-	0.1	1.0	Test both with/without VGG
# 128×128	False	-	0.1-0.2	1.0	Native resolution works best
# 256×256	True	224	0.1-0.3	1.0	Standard configuration
# 512×512	True	384	0.2-0.4	1.0	Larger target preserves details
# 1024×1024	True	512	0.3-0.5	1.0	Or use multi-scale (see below)