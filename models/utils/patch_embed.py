from torch import nn


class PatchEmbed(nn.Module):
    """Convert image tensor to patch embeddings."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x = [B, C, H, W]
        x = self.proj(x)
        # x = [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)
        # x = [B, (H/patch_size) * (W/patch_size), embed_dim]
        # each patch is a vector of embed_dim dimensions
        return x
