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
        raise ValueError('num_heads must divide by kv_heads')

    key = key.repeat_interleave(num_heads // kv_heads, dim=2)
    key = key.permute(0, 2, 1, 3)
    value = value.repeat_interleave(num_heads // kv_heads, dim=2)
    value = value.permute(0, 2, 1, 3)

    query = query.permute(0, 2, 1, 3)

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (embed_dim ** 0.5)

    if is_causal:
        mask = torch.triu(torch.ones(seq_len, kv_seq_len), diagonal=1).bool()
        attn_scores.masked_fill_(mask.to(attn_scores.device), float('-inf'))

    attn_weights = F.softmax(attn_scores, dim=-1)

    out_vanilla = torch.matmul(attn_weights, value)

    out_vanilla = out_vanilla.permute(0, 2, 1, 3)

    if need_weights:
        return out_vanilla, attn_weights
    else:
        return out_vanilla, None
