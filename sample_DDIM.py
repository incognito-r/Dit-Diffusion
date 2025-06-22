import os
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import DiTTransformer2DModel
from omegaconf import OmegaConf
from utils.ema import create_ema_model
from utils.metrics.gpu import init_nvml, gpu_info

@torch.no_grad()
def main():
    config = OmegaConf.load("configs/train_config_256.yaml")
    sample_config = config.sampling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    # === Sampling ===
    output_dir = sample_config.dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "samples_grid.png")
    num_samples = sample_config.get("num_samples", 4)  # Default to 4 samples if not specified
    steps = sample_config.get("steps", 50)  # Default to 50 steps if not specified

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

    # === Load EMA model ===
    ema_model, ema = create_ema_model(model, beta=config.training.ema_beta, step_start_ema=config.training.step_start_ema)
    
    # checkpoint #
    ckpt_dir = config.checkpoint.path
    ema_ckpt_name = config.checkpoint.ema_ckpt_name
    # ckpt_name = config.checkpoint.ckpt_name

    ema_ckpt_path = os.path.join(ckpt_dir, ema_ckpt_name)
    # ckpt_path = "checkpoints/ema_epoch_1.pth"  # Update if needed

    ema_model.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
    ema_model.eval()

    # === DDIM Scheduler ===
    scheduler = DDIMScheduler(
        num_train_timesteps=config.scheduler.timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        beta_schedule="linear"
    )
    
      # You can change this to 25 / 100 / 250 etc.
    scheduler.set_timesteps(steps)

    latent_shape = (num_samples, config.model.latent_dim, config.model.img_size, config.model.img_size)
    x = torch.randn(latent_shape).to(device)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling")):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            class_labels = torch.zeros_like(t_batch)
            noise_pred = ema_model(x, timestep=t_batch, class_labels=class_labels).sample

        x = scheduler.step(noise_pred, t, x).prev_sample

    # === Decode latents ===
    imgs = vae.decode(x / 0.18215).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # [-1, 1] → [0, 1]

    # === Save Grid ===
    save_image(make_grid(imgs, nrow=5), output_path)
    print(f"✅ Samples saved to {output_path}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
