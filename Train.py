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
    print(f"Configuration loaded: {OmegaConf.to_yaml(config)}")
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
    
    # Optimizer for DiT model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    # losses
    MSE_LOSS_Dit = torch.nn.MSELoss()
    LPIPS_LOSS = lpips.LPIPS(net='vgg').to(device).eval()

    print("Models, optimizers, losses initialized successfully.")
    #===================================================================

    # === Load data ===
    dataloader, _ = CelebAloader(data_config=config.data, train_config=config.training)

    print(f"Dataset size: {len(dataloader.dataset)} images")
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}, Device: {batch.device}")

    # === Load checkpoint ===
    checkpoint_dir = config.checkpoint.path
    dit_ckpt_path = os.path.join(checkpoint_dir, config.checkpoint.ckpt_name)
    # ckpt_path = "checkpoints/dit_diffusion_ckpt_256.pth"
    # ema_ckpt_path = "checkpoints/dit_diffusion_ema_ckpt_256.pth"
    
    start_epoch, best_loss = load_training_state(dit_ckpt_path, model, optimizer, device)
    print(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")

    # ===== Training Loop =====
    warmup_ep = config.training.warmup_epochs
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

            # ---- Forward Diffusion ----
            t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device)

            noise = torch.randn_like(latents)
            x_t = scheduler.add_noise(latents, noise, t)

            # ---- Noise Prediction ----
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                dummy_class_labels = torch.zeros(latents.shape[0], dtype=torch.long, device=device)
                noise_pred = model(x_t, timestep=t, class_labels=dummy_class_labels).sample
                
                # ==== Loss Calculation ===
                mse_loss = MSE_LOSS_Dit(noise_pred, noise) / config.training.grad_accum_steps

                # -----
                if epoch+1 < warmup_ep: # No other loss
                    lpips_loss = 0.0
                    lpips_weight = 0.0

                else: # Compute LPIPS loss or other losses
                    pred_x0 = scheduler.step(noise_pred, t[0].item(), x_t).pred_original_sample
                    pred_rgb = vae.decode(pred_x0 / 0.18215).sample.clamp(-1, 1)

                    lpips_loss = LPIPS_LOSS(pred_rgb, images).mean()
                    
                    # gradually increase loss weights
                    if epoch+1 < 50:
                        lpips_weight = 0.05 * (epoch+1 - warmup_ep) / (30 - warmup_ep)
                    else:
                        lpips_weight = 0.1
                # -----

                # Total loss
                total_loss= mse_loss + lpips_weight * lpips_loss
                # ======================
              
            loss = total_loss

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
            checkpoint_path=dit_ckpt_path,
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