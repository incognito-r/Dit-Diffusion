import os
import random
import numpy as np
import torch
from datetime import datetime
import csv

def save_metadata(checkpoint_dir, epoch, checkpoint_path, avg_loss, best_loss):
    # Use different files for regular and best checkpoints
    metadata_file = os.path.join(checkpoint_dir, "training_metadata.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create metadata entry
    metadata_entry = {
        'timestamp': timestamp,
        'epoch': epoch,
        'checkpoint_path': checkpoint_path,
        'avg_loss': f"{avg_loss:.6f}",
        'best_loss': f"{best_loss:.6f}"
    }

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(metadata_file)
    
    # Append to metadata file
    with open(metadata_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metadata_entry)

def load_training_state(checkpoint_path, model, optimizer, discriminator=None, optimizer_d=None, 
                      lr_scheduler=None, device='cpu'):
    """Load training state with discriminator support"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        return 0, float('inf')
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"✅ Resuming from checkpoint: {checkpoint_path}")
    
    # Initialize best_loss
    best_loss = ckpt.get('best_loss', float('inf'))
    
    # Load core components
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print("Loaded model and optimizer")
    # Handle discriminator (new in v2)
    if discriminator:
        discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        if optimizer_d:
            optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
            print("Loaded discriminator and optimizer")

    elif not discriminator:
        print("⚠️ No discriminator found - initializing new one")
        # Initialize discriminator weights here if needed
    
    # Handle scheduler
    if lr_scheduler and "lr_scheduler_state_dict" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        print("Loaded learning rate scheduler")
    
    start_epoch = ckpt["epoch"] + 1  # Start from NEXT epoch
    print(f"Resuming at epoch {start_epoch}, previous loss: {ckpt['loss']:.4f}")
    
    return start_epoch, best_loss

def save_training_state(checkpoint_path, epoch, model, optimizer, avg_loss, best_loss, 
                      discriminator=None, optimizer_d=None, lr_scheduler=None):
    """Save training state with discriminator support"""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "best_loss": best_loss,
    }
    
    # Add discriminator if available
    if discriminator:
        ckpt["discriminator_state_dict"] = discriminator.state_dict()
    if optimizer_d:
        ckpt["optimizer_d_state_dict"] = optimizer_d.state_dict()
    
    # Add scheduler if available
    if lr_scheduler:
        ckpt["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    
    torch.save(ckpt, checkpoint_path)
    print(f"💾 Saved checkpoint at epoch {epoch+1}")
    



    # ===== Save metadata after saving the checkpoint =====
    # Save metadata. This will create or append to a CSV file in the same directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    save_metadata(checkpoint_dir, epoch+1, checkpoint_path, avg_loss, best_loss if best_loss is not None else avg_loss)