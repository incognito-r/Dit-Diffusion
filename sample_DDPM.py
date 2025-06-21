import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.models import DiTTransformer2DModel
from omegaconf import OmegaConf
from utils.ema import create_ema_model
from utils.metrics.gpu import init_nvml, gpu_info

@torch.no_grad()
def main():
    config = OmegaConf.load("configs/train_config_256.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    handle = init_nvml()
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    
    # === Load VAE ===
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

    # === Load DiT Model ===
    model = DiTTransformer2DModel(
        in_channels=config.model.latent_dim,
        num_attention_heads=config.model.num_heads,
        attention_head_dim=config.model.attn_head_dim,
        num_layers=config.model.depth,
        sample_size=config.model.img_size // config.model.patch_size,
        patch_size=config.model.patch_size,
    ).to(device)

    # === Load EMA model wrapper ===
    ema_model, ema = create_ema_model(model, beta=config.training.ema_beta, step_start_ema=config.training.step_start_ema)
    
    # === Load checkpoint ===
    ckpt_path = "checkpoints/dit_diffusion_ema_ckpt_256.pth"  # change if needed
    ema_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    ema_model.eval()

    # === Load scheduler ===
    scheduler = DDPMScheduler(
        num_train_timesteps=config.scheduler.timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        beta_schedule="linear"
    )

    # === Sampling ===
    batch_size = 4
    latent_shape = (batch_size, config.model.latent_dim, config.model.img_size, config.model.img_size)
    x = torch.randn(latent_shape).to(device)

    pbar = tqdm(reversed(range(scheduler.num_train_timesteps)), desc="Sampling", total=scheduler.num_train_timesteps)

    for t in pbar:
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            class_labels = torch.zeros_like(t_batch)
            noise_pred = ema_model(x, timestep=t_batch, class_labels=class_labels).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
        pbar.set_postfix(mem=gpu_info(handle))

    # === Decode latents to image ===
    imgs = vae.decode(x / 0.18215).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # [-1, 1] → [0, 1]

    # === Save or Display ===
    os.makedirs("output/samples", exist_ok=True)
    save_image(make_grid(imgs, nrow=2), "output/samples/sample_grid.png")
    print("✅ Sample saved to output/samples/sample_grid.png")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
