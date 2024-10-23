import torch
import torch.nn.functional as F
import torchtune


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    batch_size, seq_length, dim_per_head = queries.size()
    scores = torch.bmm(queries, keys.transpose(1, 2))
    scores /= dim_per_head**0.5
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.double)
    if queries.dtype == torch.float32:
        attention_weights = attention_weights.type(torch.float32)
    return torch.bmm(attention_weights, values)


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    queries = queries.double()
    keys = keys.double()
    values = values.double()
    projection_matrix = projection_matrix.double()

    batch_size, n_heads, seq_length, dim_per_head = queries.shape
    outputs = []
    for i in range(n_heads):
        head_queries = queries[:, i]  # (BATCH_SIZE, SEQ_LENGTH, DIM_PER_HEAD)
        head_keys = keys[:, i]
        head_values = values[:, i]
        head_output = F.scaled_dot_product_attention(head_queries, head_keys, head_values)
        outputs.append(head_output)
    concatenated_output = torch.cat(outputs, dim=2)  # (BATCH_SIZE, SEQ_LENGTH, DIM_PER_HEAD * N_HEADS)
    concatenated_output = concatenated_output.matmul(projection_matrix.T)
    return concatenated_output.float()


def compute_rotary_embeddings(x) -> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    batch_size, seq_length, n_heads, dim_per_head = x.size()
    rpe = torchtune.modules.RotaryPositionalEmbeddings(dim=dim_per_head, max_seq_len=seq_length)
    return rpe(x)
