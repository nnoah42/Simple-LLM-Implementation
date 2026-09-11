import torch
from torch import nn

class InputEmbedding(nn.Module):
    """
    InputEmbedding module for token and positional embeddings. Includes methods for 
    initialization and forward pass.
    """
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        embedding_dim: int
    ) -> None:
        """Initialize token and positional embeddings tables."""
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim
        )
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Gets corresponding token vectors from token ids and adds positional embeddings.

        Args:
            token_ids: Tensor of shape (batch_size, sequence_length).

        Returns: 
            Tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        batch_size, sequence_length = token_ids.shape
        token_vectors = self.token_embedding(token_ids)
        positions = torch.arange(
            sequence_length
        )
        position_vectors = self.position_embedding(positions)
        return token_vectors + position_vectors