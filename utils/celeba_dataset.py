import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, config):
        self.img_dir = config.path
        self.img_size = config.image_size 
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]) # In one folder
        # Include all .jpg and .png files from subdirectories
        # self.filenames = sorted([
        #     os.path.join(root, f)
        #     for root, _, files in os.walk(self.img_dir)
        #     for f in files
        #     if f.lower().endswith('.jpg') or f.lower().endswith('.png')
        # ])

        self.T = transforms.Compose([
            transforms.Resize(self.img_size),  # Resize to the specified size
            transforms.CenterCrop(self.img_size), # Ensure the image size
            transforms.ToTensor(),  # Converts to [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        # img_path = self.filenames[index]
        # img_path = os.path.join(self.img_dir, self.filenames[index])
        rgb_img = Image.open(img_path).convert('RGB')
        img_tensor = self.T(rgb_img)
        return img_tensor

#------------------------------------------------------------------------------
def CelebAloader(data_config, train_config):
    val_split = train_config.get('validation_split', 0)  
    batch_size = train_config.get('batch_size', 32) 
    num_workers = train_config.get('num_workers', 4) 

    full_dataset = CelebADataset(config=data_config)
    total_len = len(full_dataset)

    indices = list(range(total_len))
    split = int(total_len * (1 - val_split))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader