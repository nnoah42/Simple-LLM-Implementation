# Simple LLM Implementation

Small, readable building blocks for learning how a GPT-style language model
works. The code follows concepts from *Build a Large Language Model (From
Scratch)* by Sebastian Raschka.

## Setup

```bash
python -m pip install torch tiktoken
python download_data.py
```

The project is currently intended to run on a CPU. Minimal device-handling
changes are needed to run it with CUDA.

## Project structure

- `gpt_dataset.py`: converts text into input and target token sequences.
- `dataloader.py`: creates PyTorch batches from the dataset.
- `tokenizer.py`: a small vocabulary-based tokenizer for learning purposes.
- `download_data.py`: downloads the sample training text.
