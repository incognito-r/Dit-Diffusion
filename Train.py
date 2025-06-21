import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, DiTTransformer2DModel
from models.discriminator import PatchDiscriminator
from utils.ema import create_ema_model
from utils.checkpoint import save_training_state, load_training_state
from utils.celeba_dataset import CelebAloader
from utils.metrics.gpu import init_nvml, gpu_info
from omegaconf import OmegaConf
import lpips

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
    print(f"Configuration loaded: {config}")
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
    ).to(device) # Dit Model

    # === Load Patch Discriminator ===
    discriminator = PatchDiscriminator().to(device)

    # === Load noise scheduler from diffusers ===
    scheduler = DDPMScheduler(
        num_train_timesteps=config.scheduler.timesteps,
        beta_start=config.scheduler.beta_start,
        beta_end=config.scheduler.beta_end,
        beta_schedule="linear",
    )

    # EMA model
    ema_model, ema = create_ema_model(model, beta=config.training.ema_beta, step_start_ema=config.training.step_start_ema)

    # Optimizer for DiT model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    # Optimizer for Discriminator
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=config.training.lr * 0.5)  # Lower LR for D

    # === losses ====
    MSE_LOSS = torch.nn.MSELoss()
    MSE_LOSS_D = torch.nn.MSELoss()
    LPIPS_LOSS = lpips.LPIPS(net='vgg').to(device).eval()

    print("Models and optimizers initialized successfully.")
    #===================================================================

    # === Load data ===
    dataloader, _ = CelebAloader(data_config=config.data, train_config=config.training)

    print("Models and optimizers initialized successfully.")
    print(f"Dataset size: {len(dataloader.dataset)} images")
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}, Device: {batch.device}")

    # === Load checkpoint ===
    checkpoint_dir = config.checkpoint.path
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_path = os.path.join(checkpoint_dir, config.checkpoint.ckpt_name)
    ema_ckpt_path = os.path.join(checkpoint_dir, config.checkpoint.ema_ckpt_name)

    # ckpt_path = "checkpoints/dit_diffusion_ckpt_256.pth"
    # ema_ckpt_path = "checkpoints/dit_diffusion_ema_ckpt_256.pth"

    start_epoch, best_loss = load_training_state(ckpt_path, model, optimizer, discriminator=discriminator, optimizer_d=optimizer_d, device=device)
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
            t = t[0] if len(t) == 1 else t  # Ensure proper shape
            noise = torch.randn_like(latents)
            x_t = scheduler.add_noise(latents, noise, t)

            # ---- Noise Prediction ----
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):

                # --- Generator (DiT) Forward ---
                class_labels = torch.zeros(latents.shape[0], dtype=torch.long, device=device) # dummy labels for now
                noise_pred = model(x_t, timestep=t, class_labels=class_labels).sample

                pred_x0 = scheduler.step(
                    model_output=noise_pred, 
                    timestep=t[0].item(), # use t[0].item() for batch processing
                    sample=x_t).pred_original_sample
                
                pred_rgb = vae.decode(pred_x0 / 0.18215).sample.clamp(-1, 1)

                # --- Losses ---
                mse_loss = MSE_LOSS(noise_pred, noise) / config.training.grad_accum_steps
                lpips_loss = LPIPS_LOSS(pred_rgb, images).mean()
                fake_logits = discriminator(pred_rgb)
                g_loss = MSE_LOSS_D(fake_logits, torch.ones_like(fake_logits)) * 0.05  # Weighted
                total_loss= mse_loss + 0.3 * lpips_loss + g_loss  # Generator total loss 

            # ---- Backward Pass ----
            scaler.scale(total_loss).backward()

            # ---- Discriminator Update ----
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                # Real images
                real_logits = discriminator(images)
                d_loss_real = MSE_LOSS_D(real_logits, torch.ones_like(real_logits))
                
                # Fake images (detach gradients)
                fake_logits = discriminator(pred_rgb.detach()) # Critical: detach()
                d_loss_fake = MSE_LOSS_D(fake_logits, torch.zeros_like(fake_logits))
                
                d_loss = (d_loss_real + d_loss_fake) * 0.5 
            
            # Backward Pass (Discriminator)
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step() # Update D (no gradient accumulation)

            # ---- Gradient Accumulation ----
            if (batch_idx + 1) % config.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.step_ema(ema_model, model)
                optimizer.zero_grad()

            # ---- Progress Tracking ----
            running_loss += total_loss.item() * config.training.grad_accum_steps  # Scale back for reporting
            avg_loss = running_loss / (batch_idx + 1)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
            pbar.set_postfix(avg_loss=avg_loss, mem=gpu_info(handle))

        # ---- Checkpointing ----
        save_training_state(
            checkpoint_path=ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            discriminator=discriminator,
            optimizer_d=optimizer_d,
            avg_loss=avg_loss,
            best_loss=best_loss,
        )
        
        # Save EMA model
        torch.save(ema_model.state_dict(), ema_ckpt_path)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Training Completed!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    main()