import torch
from torch import nn

from embedding import InputEmbedding
from transformer import TransformerBlock


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_heads,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.embedding = InputEmbedding(
            vocab_size,
            context_length,
            embedding_dim,
        )
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_normalization = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    def forward(self, token_ids):
        _, sequence_length = token_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length cannot exceed {self.context_length}"
            )

        hidden_states = self.embedding(token_ids)
        hidden_states = self.transformer_blocks(hidden_states)
        hidden_states = self.final_normalization(hidden_states)
        return self.output_projection(hidden_states)
