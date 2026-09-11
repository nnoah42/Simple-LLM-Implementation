import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
	def __init__(self, embedding_dim, num_heads, dropout=0.0):
		super().__init__()

		if embedding_dim % num_heads != 0:
			raise ValueError("embedding_dim must be divisible by num_heads")

		self.num_heads = num_heads
		self.head_dim = embedding_dim // num_heads
		self.query_key_value = nn.Linear(embedding_dim, 3 * embedding_dim)
		self.output_projection = nn.Linear(embedding_dim, embedding_dim)
		self.attention_dropout = nn.Dropout(dropout)
		self.output_dropout = nn.Dropout(dropout)

	def forward(self, inputs):
		batch_size, sequence_length, embedding_dim = inputs.shape

		query, key, value = self.query_key_value(inputs).chunk(3, dim=-1)

		query = query.view(
			batch_size,
			sequence_length,
			self.num_heads,
			self.head_dim,
		).transpose(1, 2)
		key = key.view(
			batch_size,
			sequence_length,
			self.num_heads,
			self.head_dim,
		).transpose(1, 2)
		value = value.view(
			batch_size,
			sequence_length,
			self.num_heads,
			self.head_dim,
		).transpose(1, 2)

		attention_scores = query @ key.transpose(-2, -1)
		attention_scores = attention_scores / math.sqrt(self.head_dim)

		causal_mask = torch.triu(
			torch.ones(
				sequence_length,
				sequence_length,
				device=inputs.device,
				dtype=torch.bool,
			),
			diagonal=1,
		)
		attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

		attention_weights = torch.softmax(attention_scores, dim=-1)
		attention_weights = self.attention_dropout(attention_weights)
		output = attention_weights @ value

		output = output.transpose(1, 2).contiguous()
		output = output.view(batch_size, sequence_length, embedding_dim)
		return self.output_dropout(self.output_projection(output))


class FeedForward(nn.Module):
	def __init__(self, embedding_dim, dropout=0.0):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(embedding_dim, 4 * embedding_dim),
			nn.GELU(),
			nn.Linear(4 * embedding_dim, embedding_dim),
			nn.Dropout(dropout),
		)

	def forward(self, inputs):
		return self.network(inputs)


class TransformerBlock(nn.Module):
	def __init__(self, embedding_dim, num_heads, dropout=0.0):
		super().__init__()
		self.normalization_before_attention = nn.LayerNorm(embedding_dim)
		self.attention = CausalSelfAttention(
			embedding_dim,
			num_heads,
			dropout,
		)
		self.normalization_before_feed_forward = nn.LayerNorm(embedding_dim)
		self.feed_forward = FeedForward(embedding_dim, dropout)

	def forward(self, inputs):
		attention_input = self.normalization_before_attention(inputs)
		inputs = inputs + self.attention(attention_input)

		feed_forward_input = self.normalization_before_feed_forward(inputs)
		return inputs + self.feed_forward(feed_forward_input)
