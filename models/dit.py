import torch
import torch.nn as nn
import torch.nn.functional as F

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, C] where N = H*W
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DiT(nn.Module):
    def __init__(self, 
                 in_channels=4,      # latent channels from VAE
                 patch_size=1,       # treat each pixel as patch
                 hidden_size=768,    # transformer dimension
                 depth=12,           # number of DiT blocks
                 num_heads=12, 
                 img_size=64, 
                 dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.input_proj = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_size))
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(dim=hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Conv2d(hidden_size, in_channels, kernel_size=1)

    def forward(self, x, t):
        # x: [B, 4, 64, 64] → latents from VAE
        B, C, H, W = x.shape

        x = self.input_proj(x)           # → [B, hidden, H, W]
        x = x.flatten(2).transpose(1, 2) # → [B, N, hidden]
        x = x + self.pos_embed           # add positional embedding

        # Time embedding
        temb = self.sinusoidal_embedding(t)         # [B, hidden]
        temb = self.time_embed(temb).unsqueeze(1)   # [B, 1, hidden]
        x = x + temb                                # add timestep conditioning

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.transpose(1, 2).view(B, -1, H, W)  # [B, hidden, H, W]
        return self.output_proj(x)               # [B, 4, 64, 64]

    def sinusoidal_embedding(self, t, dim=768):
        device = t.device
        half_dim = dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb
