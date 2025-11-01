from .architecture import ViTConfig, VisionTransformer


def vit_small(num_classes: int, image_size: int = 224) -> VisionTransformer:
    """Construct a ViT-Small model."""
    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=num_classes,
        stochastic_depth=0.1,
    )
    return VisionTransformer(config)
