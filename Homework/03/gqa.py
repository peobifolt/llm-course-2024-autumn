import torch
import torch.nn.functional as F


def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """
    batch_size, seq_len, num_heads, embed_dim = query.shape
    kv_seq_len = key.shape[1]
    kv_heads = key.shape[2]
    if num_heads % kv_heads:
        raise ValueError('num_heads must divides by kv_heads')

    key = key.repeat_interleave(num_heads // kv_heads, dim=2)
    key = key.permute(0, 2, 1, 3)
    value = value.repeat_interleave(num_heads // kv_heads, dim=2)
    value = value.permute(0, 2, 1, 3)

    query = query.permute(0, 2, 1, 3)

    out_vanilla = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

    out_vanilla = out_vanilla.permute(0, 2, 1, 3)
    return out_vanilla, None
