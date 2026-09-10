# Simple LLM Implementation

Small, readable building blocks for learning how a GPT-style language model
works. The code follows concepts from *Build a Large Language Model (From
Scratch)* by Sebastian Raschka.

## Setup

```bash
python -m pip install torch tiktoken
python download_data.py
```

## Create a dataloader

```python
from dataloader import create_dataloader

with open("the-verdict.txt", encoding="utf-8") as file:
	text = file.read()

dataloader = create_dataloader(text, batch_size=4, max_length=256, stride=128)
inputs, targets = next(iter(dataloader))
```

`inputs` contains token sequences. `targets` contains the same sequences
shifted by one token, which is the training target for next-token prediction.

## Project structure

- `gpt_dataset.py`: converts text into input and target token sequences.
- `dataloader.py`: creates PyTorch batches from the dataset.
- `tokenizer.py`: a small vocabulary-based tokenizer for learning purposes.
- `download_data.py`: downloads the sample training text.
