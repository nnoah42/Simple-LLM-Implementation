# NOT CURRENTLY USED. TIKTOKEN IS USED INSTEAD.
import re

class SimpleTokenizer:
    def __init__(self, vocabulary):
        self.str_to_int = vocabulary
        self.int_to_str = {index: token for token, index in vocabulary.items()}

    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        tokens = [token if token in self.str_to_int else "<|unk|>" for token in tokens]
        return [self.str_to_int[token] for token in tokens]

    def decode(self, token_ids):
        text = " ".join(self.int_to_str[token_id] for token_id in token_ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)