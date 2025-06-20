from os import path, makedirs
from re import split

import numpy as np
import torch

datafile = "./data/vocab.txt"


def encode_tokens_one_hot_encoding(tokens: [int], context_size: int = 1024, vocab_size: int = 1024) -> torch.Tensor:
    encoded_arr = np.zeros((max(context_size, len(tokens)), vocab_size))
    encoded_arr[np.arange(len(tokens)), tokens] = 1
    return torch.tensor(encoded_arr, dtype=torch.float32)


class Tokenizer:
    def __init__(self):
        super().__init__()
        self.tokens = [
            "[SOS]",
            "[EOS]"
        ]
        self.tokens_updated = False
        if path.exists(datafile):
            with open(datafile, "r") as f:
                self.tokens.extend([line.strip() for line in f.readlines()])

    def tokenize(self, text):
        text = self.normalise(text)
        tokenized_list = [0]
        for word in Tokenizer.split(text):
            if word not in self.tokens:
                self.tokens_updated = True
                self.tokens.append(word)
            tokenized_list.append(self.tokens.index(word))

        tokenized_list.append(1)
        return tokenized_list

    @staticmethod
    def normalise(text):
        return text.lower()

    @staticmethod
    def split(text):
        return list(filter(None, split(r'\s+|([^0-9a-zA-Z\s])', text)))

    def textual(self, tokenized_array):
        return " ".join([self.tokens[token_idx] for token_idx in tokenized_array])

    def decode(self, tokenized_array):
        return " ".join([self.tokens[token_idx] for token_idx in tokenized_array if token_idx > 1])

    def save(self):
        if not self.tokens_updated:
            return

        makedirs(path.dirname(datafile), exist_ok=True)
        with open(datafile, "w") as f:
            f.writelines([word + "\n" for word in self.tokens[2:]])


######################################################################################
#                                     TESTS
######################################################################################
def test_tokenizer():
    test_text = "Hello World!"
    test_token_len = 4
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(test_text)
    assert len(
        tokens) == test_token_len, f"Expected {test_token_len} token for: {test_text} but generated: {len(tokens)}."


def test_save_dictionary():
    test_text = "Hello World!"
    tokenizer = Tokenizer()
    tokenizer.tokenize(test_text)
    tokenizer.save()
    assert path.exists(datafile), f"Tokenizer.save() didn't generate expected file: {datafile}"


def test_functions():
    test_text = "Hello World!"
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(test_text)
    result = tokenizer.textual(tokens)
    assert result == "[SOS] hello world ! [EOS]", f"Textual representation: {result}, didn't match expected."

    result = tokenizer.decode(tokens)
    assert result == "hello world !", f"Textual representation: {result}, didn't match expected."


def test_one_hot_encoded_shape():
    test_text = "How are you today?"
    tokenizer = Tokenizer()
    out = encode_tokens_one_hot_encoding(tokenizer.tokenize(test_text))

    assert out.shape == torch.Size(
        [1024, 1024]
    ), f"Expected an output tensor of size (1024, 1024), got {out.shape}"
