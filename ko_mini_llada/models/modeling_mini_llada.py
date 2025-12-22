# modeling_llada.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from ko_mini_llada.models.configuration_mini_llada import MiniLLaDAConfig

# RoPE (Rotary Positional Embedding)
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

class Attention(nn.Module):
    """
    Multi-Head Attention module with Rotary Positional Embeddings.

    This module implements a multi-head attention mechanism, incorporating rotary
    positional embeddings (RoPE) for query and key tensors. It uses bidirectional
    attention as required by the LLaDA architecture.

    Args:
        dim (int): The input and output dimension of the module.
        heads (int): The number of attention heads.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.n_heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        """
        Forward pass for the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            freqs_cis (torch.Tensor): Precomputed rotary frequencies.
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, L, D).
        """
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        freqs_cis = freqs_cis.to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis[:L])

        # LLaDA: is_causal=False (Bidirectional Attention)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        x = attn_output.transpose(1, 2).reshape(B, L, D)
        x = self.o_proj(x)
        return x

class Block(nn.Module):
    """
    Transformer block consisting of multi-head attention and a feed-forward network.

    This block applies pre-normalization (RMSNorm), followed by multi-head attention,
    a residual connection, another pre-normalization, a feed-forward network with
    SwiGLU activation, and a final residual connection.

    Args:
        dim (int): The dimension of the input and output.
        heads (int): The number of attention heads.
        intermediate_size (int): The intermediate size of the feed-forward network.
    """
    def __init__(self, dim, heads, intermediate_size):
        super().__init__()
        self.mha = Attention(dim, heads) # Multi-Head Attention
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        # Feed-Forward Network with Gated Linear Unit
        self.gate_up_proj = nn.Linear(dim, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            freqs_cis (torch.Tensor): Precomputed rotary frequencies.
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, L, D).
        """
        # Multi-Head Attention
        residual = x
        x = self.norm1(x)
        x = self.mha(x, freqs_cis, mask)
        x = residual + x

        # Feed-Forward Network
        residual = x
        x = self.norm2(x)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        x = residual + x
        return x

class Transformer(nn.Module):
    """
    The core Transformer model for Mini-LLaDA.

    This class stacks multiple Transformer blocks to form the main network.
    It includes token embeddings, a series of Transformer blocks, a final
    normalization layer, and a linear head to project to the vocabulary size.

    Args:
        vocab_size (int): The size of the vocabulary.
        dim (int): The embedding dimension.
        depth (int): The number of Transformer blocks.
        heads (int): The number of attention heads.
        intermediate_size (int): The intermediate size of the feed-forward networks.
        max_seq_len (int, optional): The maximum sequence length. Defaults to 2048.
    """
    def __init__(self, vocab_size, dim, depth, heads, intermediate_size, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim) # Token Embedding
        self.layers = nn.ModuleList([
            Block(dim, heads, intermediate_size) for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        freqs_cis = precompute_freqs_cis(dim // heads, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for the Transformer model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (B, L).
            attention_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens.
                Shape (B, L). Defaults to None.

        Returns:
            torch.Tensor: Output logits of shape (B, L, vocab_size).
        """
        x = self.embed(input_ids)
        
        mask = None
        if attention_mask is not None:
            # Create mask for sdpa: (B, 1, 1, L). True indicates position to mask (padding).
            mask = (attention_mask == 0).view(x.shape[0], 1, 1, x.shape[1])

        for layer in self.layers:
            x = layer(x, self.freqs_cis, mask)
        return self.head(self.norm(x))

class MiniLLaDA(PreTrainedModel):
    """
    The Mini-LLaDA model, a Transformer-based model for masked language modeling
    inspired by diffusion models.

    This model is designed for pre-training using a diffusion-like noising process
    where a variable number of tokens are masked and the model learns to predict
    the original tokens. It can be used for both training (with labels) and
    inference (without labels).

    Args:
        config (MiniLLaDAConfig): The configuration object for the model.
    """
    config_class = MiniLLaDAConfig

    def __init__(self, config: MiniLLaDAConfig):
        super().__init__(config)
        
        # 1. load backbone model
        self.network = Transformer(
            vocab_size=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.head,
            intermediate_size=config.intermediate_size,
            max_seq_len=config.max_seq_len
        )
        self.mask_token_id = config.mask_token_id

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for the Mini-LLaDA model.

        If `labels` are provided, the model operates in training/evaluation mode,
        performing the diffusion forward process, running the noised input through
        the network, and computing the loss.

        If `labels` are not provided, the model operates in inference mode, simply
        passing the `input_ids` through the network to get logits.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (B, L).
            attention_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens.
                Shape (B, L). Defaults to None.
            labels (torch.Tensor, optional): Labels for computing the loss. In SFT, this is used
                to identify which parts of the sequence to mask (assistant's response).
                Shape (B, L). Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            transformers.modeling_outputs.MaskedLMOutput: An output object containing the
                loss (if labels are provided) and logits.
        """
        # 1. Training and Evaluation Mode
        if labels is not None:
            # Diffusion Forward Process
            t, noisy_x, mask_indices = self.forward_process(input_ids, labels)
            
            # Reverse Process
            # network outputs: MaskedLMOutput (logits, hidden_states, etc.)
            outputs = self.network(input_ids=noisy_x, attention_mask=attention_mask)
            
            # Compute Loss
            loss = self.compute_diffusion_loss(outputs, input_ids, mask_indices, attention_mask)
            
            return MaskedLMOutput(loss=loss, logits=outputs)

        # 2. Inference Mode
        else:
            outputs = self.network(input_ids=input_ids, attention_mask=attention_mask)
            return MaskedLMOutput(logits=outputs)

    def forward_process(self, input_ids, labels=None):
        """
        Simulates the diffusion forward process by noising the input sequence.

        A random timestep `t` is sampled for each sequence in the batch, which
        determines the probability of a token being masked. Tokens are replaced
        with a `mask_token_id`. During supervised fine-tuning (SFT), masking is
        restricted to the assistant's response part of the sequence, identified
        by `labels != -100`.

        Args:
            input_ids (torch.Tensor): The original input token IDs of shape (B, L).
            labels (torch.Tensor, optional): Labels used to restrict masking during SFT.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - t (torch.Tensor): The sampled timesteps for each sequence (shape B).
                - noisy_x (torch.Tensor): The noised input IDs with masks (shape B, L).
                - mask_indices (torch.Tensor): A boolean tensor indicating which tokens
                  were masked (shape B, L).
        """
        B, L = input_ids.shape
        device = input_ids.device

        t = torch.rand(B, device=device)
        mask_probs = t.unsqueeze(1).expand(B, L)

        if labels is not None:
            train_mask = (labels != -100).float() 
            mask_probs = mask_probs * train_mask  # Make the probabilities zero where labels == -100, which is user message parts.

        random_matrix = torch.rand(B, L, device=device)
        mask_indices = (random_matrix < mask_probs)

        noisy_x = torch.where(mask_indices, self.mask_token_id, input_ids)
        return t, noisy_x, mask_indices

    def compute_diffusion_loss(self, logits, input_ids, mask_indices, attention_mask):
        """
        Computes the cross-entropy loss for the masked language modeling task.

        The loss is calculated only for the positions that were masked during the
        forward process. It also respects the attention mask to avoid computing
        loss on padding tokens.

        Args:
            logits (torch.Tensor): The model's output logits of shape (B, L, V).
            input_ids (torch.Tensor): The original input token IDs of shape (B, L).
            mask_indices (torch.Tensor): A boolean tensor indicating masked positions (shape B, L).
            attention_mask (torch.Tensor): The attention mask for the input (shape B, L).

        Returns:
            torch.Tensor: The computed cross-entropy loss.
        """
        B, L, V = logits.shape
        logits_flat = logits.view(-1, V)
        target_flat = input_ids.view(-1)
        
        # We compute loss only on tokens that were masked AND are not padding.
        final_loss_mask = mask_indices.view(-1)
        if attention_mask is not None:
            final_loss_mask = final_loss_mask & attention_mask.view(-1).bool()
        
        target_flat = torch.where(final_loss_mask, target_flat, -100)
        return F.cross_entropy(logits_flat, target_flat, ignore_index=-100)
