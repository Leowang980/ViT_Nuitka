from torch import nn

from .drop_path import DropPath


class MLP(nn.Module):
    """Feed-forward network used inside the transformer blocks."""

    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with projection and dropout."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float,
        projection_dropout: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        bsz, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x)
        # qkv = [B, (num_tokens + 1), embed_dim * 3]
        qkv = qkv.reshape(bsz, num_tokens, 3, self.num_heads, embed_dim // self.num_heads)
        # qkv = [B, num_tokens + 1, 3, num_heads, embed_dim // num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv = [3, B, num_heads, num_tokens + 1, embed_dim // num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q = [B, num_heads, num_tokens + 1, embed_dim // num_heads]
        # k = [B, num_heads, num_tokens + 1, embed_dim // num_heads]
        # v = [B, num_heads, num_tokens + 1, embed_dim // num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(bsz, num_tokens, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderBlock(nn.Module):
    """Transformer encoder block with residual connections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        attention_dropout: float,
        projection_dropout: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, attention_dropout, projection_dropout, qkv_bias)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, projection_dropout)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
