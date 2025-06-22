import os
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from omegaconf import OmegaConf
from utils.ema import create_ema_model
from utils.metrics.gpu import init_nvml, gpu_info

@torch.no_grad()
def main():
    # Load configuration
    config = OmegaConf.load("configs/train_config_256.yaml")
    model_cfg = config.model
    sample_cfg = config.sampling

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    handle = init_nvml()

    # Prepare output
    os.makedirs(sample_cfg.dir, exist_ok=True)
    output_path = os.path.join(sample_cfg.dir, sample_cfg.filename)

    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

    # Load conditional U-Net
    model = UNet2DConditionModel(
        sample_size=model_cfg.sample_size,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        down_block_types=tuple(model_cfg.down_block_types),
        up_block_types=tuple(model_cfg.up_block_types),
        block_out_channels=tuple(model_cfg.block_out_channels),
        layers_per_block=model_cfg.layers_per_block,
        cross_attention_dim=model_cfg.cross_attention_dim,
    ).to(device)

    # EMA wrapper
    ema_model, _ = create_ema_model(model,
        beta=config.training.ema_beta,
        step_start_ema=config.training.step_start_ema
    )
    ema_ckpt = os.path.join(config.checkpoint.path, config.checkpoint.ema_ckpt_name)
    ema_model.load_state_dict(torch.load(ema_ckpt, map_location=device))
    ema_model.eval()

    # Scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=config.scheduler.timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        beta_schedule="linear"
    )
    scheduler.set_timesteps(sample_cfg.steps)

    # Load CLIP
    tokenizer = CLIPTokenizer.from_pretrained(sample_cfg.clip_model)
    text_encoder = CLIPTextModel.from_pretrained(sample_cfg.clip_model).to(device).eval()

    # Initial noise
    num_samples = sample_cfg.num_samples
    latent_shape = (num_samples, model_cfg.in_channels, model_cfg.sample_size, model_cfg.sample_size)
    x = torch.randn(latent_shape).to(device)

    # Tokenize prompt once
    prompt = "a beautiful women in a red dress"
    text_inputs = tokenizer(
        [prompt] * num_samples,
        padding="max_length",
        truncation=True,
        max_length=sample_cfg.max_length,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(**text_inputs).last_hidden_state  # [B, T, D]

    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            noise_pred = ema_model(
                x,
                timestep=t_batch,
                encoder_hidden_states=text_embeddings
            ).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
        print(f"Step {int(t)} completed | Mem: {gpu_info(handle)}")

    # Decode latents
    imgs = vae.decode(x / 0.18215).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2

    # Save grid (e.g., 10x10)
    grid = make_grid(imgs, nrow=int(num_samples**0.5))
    save_image(grid, output_path)
    print(f"✅ Samples saved to {output_path}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
