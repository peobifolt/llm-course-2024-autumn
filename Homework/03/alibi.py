import torch


def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """
    alibi = torch.zeros((num_heads, seq_len, seq_len), dtype=torch.float32)
    const = 2 if num_heads >= 8 else 4
    for head in range(num_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                alibi[head, i, j] = (j - i) / const**(head + 1)

    return alibi


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
