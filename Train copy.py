import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, DiTTransformer2DModel

from utils.ema import create_ema_model
from utils.checkpoint import save_training_state, load_training_state
from utils.celeba_dataset import CelebAloader
from utils.metrics.gpu import init_nvml, gpu_info
from omegaconf import OmegaConf

def main():
    
    torch.manual_seed(1)
    handle = init_nvml()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Enable mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    print("Mixed precision training enabled" if scaler is not None else "Mixed precision training disabled")

    # Load configuration
    config = OmegaConf.load("configs/train_config_256.yaml")
    # config = OmegaConf.load("configs/train_config_512.yaml"
    
    #==================================================================

    # === Load VAE from diffusers ===
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

    # === Load DiT from diffusers ===

    model = DiTTransformer2DModel(
        in_channels=config.model.latent_dim,
        num_attention_heads=config.model.num_heads,
        attention_head_dim=config.model.attn_head_dim,
        num_layers=config.model.depth,
        sample_size=config.model.img_size // config.model.patch_size,
        patch_size=config.model.patch_size,
    ).to(device)

    # === Load noise scheduler from diffusers ===
    scheduler = DDPMScheduler(
        num_train_timesteps=config.scheduler.timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        beta_schedule="linear",
    )

    # EMA model
    ema_model, ema = create_ema_model(model, beta=config.training.ema_beta, step_start_ema=config.training.step_start_ema)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    # losses
    criterion = torch.nn.MSELoss()

    #===================================================================

    # === Load data ===
    dataloader, _ = CelebAloader(data_config=config.data, train_config=config.training)

    print(f"Dataset size: {len(dataloader.dataset)} images")
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}, Device: {batch.device}")


    # === Load checkpoint ===
    checkpoint_path = os.path.join("checkpoints", "dit_last.pth")
    start_epoch, best_loss = load_training_state(checkpoint_path, model, optimizer, device)
    print(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")

    # ===== Training Loop =====
    for epoch in range(start_epoch, config.training.epochs):
        # ---- Memory Management ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.train()
        running_loss = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config.training.epochs}")

        for batch_idx, images in pbar:
            if batch_idx % config.training.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            images = images.to(device).float()

            # ---- VAE Encoding ----
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
            
            # ---- Forward Diffusion ----
            noise = torch.randn_like(latents)
            x_t = scheduler.add_noise(latents, noise, t)

            # ---- Noise Prediction ----
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                class_labels = torch.zeros(latents.shape[0], dtype=torch.long, device=device)
                noise_pred = model(x_t, timestep=t, class_labels=class_labels).sample

                mse_loss = criterion(noise_pred, noise) / config.training.grad_accum_steps
            loss = mse_loss

            # ---- Backward Pass ----
            scaler.scale(loss).backward()  

            # ---- Gradient Accumulation ----
            if (batch_idx + 1) % config.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.step_ema(ema_model, model)

            # ---- Progress Tracking ----
            running_loss += loss.item() * config.training.grad_accum_steps
            avg_loss = running_loss / (batch_idx + 1)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
            pbar.set_postfix(avg_loss=avg_loss, mem=gpu_info(handle))

        # ---- Checkpointing ----
        save_training_state(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            avg_loss=avg_loss,
            best_loss=best_loss,
        )
        torch.save(ema_model.state_dict(), f"checkpoints/ema_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Training Completed!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    main()