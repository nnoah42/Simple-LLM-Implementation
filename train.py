import torch
import torch.nn.functional as functional

from dataloader import create_dataloader
from model import GPTModel


def main():
	with open("the-verdict.txt", encoding="utf-8") as file:
		text = file.read()

	context_length = 64
	vocab_size = 50257
	device = "cpu"

	dataloader = create_dataloader(
		text,
		batch_size=4,
		max_length=context_length,
		stride=context_length,
	)
	model = GPTModel(
		vocab_size=vocab_size,
		context_length=context_length,
		embedding_dim=128,
		num_heads=4,
		num_layers=2,
		dropout=0.1,
	).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

	model.train()
	for epoch in range(10):
		total_loss = 0.0
		for inputs, targets in dataloader:
			inputs = inputs.to(device)
			targets = targets.to(device)

			optimizer.zero_grad()
			logits = model(inputs)
			loss = functional.cross_entropy(
				logits.reshape(-1, vocab_size),
				targets.reshape(-1),
			)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		average_loss = total_loss / len(dataloader)
		print(f"epoch {epoch + 1}: loss={average_loss:.3f}")

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"model_config": {
				"vocab_size": vocab_size,
				"context_length": context_length,
				"embedding_dim": 128,
				"num_heads": 4,
				"num_layers": 2,
				"dropout": 0.1,
			},
		},
		"model.pt",
	)
	print("saved model to model.pt")


if __name__ == "__main__":
	main()