import sys

import torch
import tiktoken

from model import GPTModel


def generate(model, token_ids, max_new_tokens, temperature=0.8, top_k=20):
	model.eval()
	with torch.no_grad():
		for _ in range(max_new_tokens):
			context = token_ids[:, -model.context_length:]
			logits = model(context)[:, -1, :] / temperature
			top_values, _ = torch.topk(logits, top_k)
			logits[logits < top_values[:, [-1]]] = float("-inf")
			probabilities = torch.softmax(logits, dim=-1)
			next_token = torch.multinomial(probabilities, num_samples=1)
			token_ids = torch.cat((token_ids, next_token), dim=1)
	return token_ids


def main():
	prompt = " ".join(sys.argv[1:]) or "It was"
	device = "cuda" if torch.cuda.is_available() else "cpu"
	checkpoint = torch.load("model.pt", map_location=device)
	model = GPTModel(**checkpoint["model_config"]).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])

	tokenizer = tiktoken.get_encoding("gpt2")
	token_ids = torch.tensor(
		[tokenizer.encode(prompt)],
		dtype=torch.long,
		device=device,
	)
	generated_ids = generate(model, token_ids, max_new_tokens=40)
	print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
	main()