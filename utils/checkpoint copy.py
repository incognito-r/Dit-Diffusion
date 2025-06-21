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

def load_training_state(checkpoint_path, model, optimizer, lr_scheduler=None, device='cpu'):

    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        return 0, float('inf')
        
    ckpt_state_dict = torch.load(checkpoint_path, map_location=device)
    print(f"✅ Resuming from checkpoint: {checkpoint_path}")
    
    # Initialize best_loss
    best_loss = ckpt_state_dict.get('best_loss', float('inf'))
    
    # Load model and optimizer states
    model.load_state_dict(ckpt_state_dict['model_state_dict'])
    optimizer.load_state_dict(ckpt_state_dict["optimizer_state_dict"])
    if lr_scheduler and "lr_scheduler_state_dict" in ckpt_state_dict:
        lr_scheduler.load_state_dict(ckpt_state_dict["lr_scheduler_state_dict"])
        print(f"Learning rate scheduler loaded from checkpoint")
    start_epoch = ckpt_state_dict["epoch"] + 1  # Start from NEXT epoch
    
    print(f"Resuming at epoch {start_epoch}, previous loss: {ckpt_state_dict['loss']:.4f}")
    
    return start_epoch, best_loss

def save_training_state(checkpoint_path, epoch, model, optimizer, avg_loss, best_loss, lr_scheduler=None):
  
    # Common checkpoint content
    ckpt_state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "best_loss": best_loss,
        # "rng_states": rng_states
    }
    if lr_scheduler is not None:
        ckpt_state_dict["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    # Save checkpoint
    torch.save(ckpt_state_dict, checkpoint_path)
    
    # Save metadata. This will create or append to a CSV file in the same directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    save_metadata(checkpoint_dir, epoch+1, checkpoint_path, avg_loss, best_loss if best_loss is not None else avg_loss)