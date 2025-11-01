from .architecture import ViTConfig, VisionTransformer


def vit_base(num_classes: int, image_size: int = 224) -> VisionTransformer:
    """Construct a ViT-Base model."""
    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        stochastic_depth=0.1,
    )
    return VisionTransformer(config)
