import diffusers
print("diffusers version:", diffusers.__version__)

from diffusers.models.attention import BasicTransformerBlock

block = BasicTransformerBlock(
    dim=320, n_heads=8, cross_attention_dim=768
)
print("attn1:", hasattr(block, "attn1"))
print("attn2:", hasattr(block, "attn2"))