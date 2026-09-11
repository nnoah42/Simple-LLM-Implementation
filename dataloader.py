import tiktoken
from torch.utils.data import DataLoader

from gpt_dataset import GPTDataset

def create_dataloader(
    text,
    batch_size=4, # targets n training examples before updating weights
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    Creates a data loader that yields batches of input and target token IDs for training.

    Args:
        text: the training text data.
        batch_size: number of samples per batch.
        max_length: max length of each input sequence.
        stride: step size for overlapping sequences.
        shuffle: whether to shuffle the dataset.
        drop_last: whether to drop the last incomplete batch.
        num_workers: number of subprocesses to use for data loading.

    Returns:
        A DataLoader instance that yields batches of (input_ids, target_ids).
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )