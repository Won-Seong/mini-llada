# RoPE (Rotary Positional Embedding)
import torch

def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0):
    """
    Precomputes the frequencies for the rotary positional embeddings.

    Args:
        dim (int): The dimension of the embeddings.
        max_len (int): The maximum sequence length.
        theta (float, optional): The theta parameter for frequency calculation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed complex frequencies.
    """  
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies rotary positional embeddings to input tensors.

    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        freqs_cis (torch.Tensor): Precomputed complex frequencies.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors with rotary embeddings applied to query and key.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Reshape freqs_cis for broadcasting. Assumes xq is (batch, n_heads, seq_len, head_dim)
    freqs_cis = freqs_cis[:xq_.shape[2]].view(1, 1, xq_.shape[2], xq_.shape[3])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)