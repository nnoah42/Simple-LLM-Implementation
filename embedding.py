import torch
from torch import nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim
            )
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape
        token_vectors = self.token_embedding(token_ids)
        positions = torch.arange(
            sequence_length
        )
        position_vectors = self.position_embedding(positions)
        return token_vectors + position_vectors