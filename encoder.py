import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (
            2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding


def positional_encoding_matrix(context_window: int):
    return torch.tensor([positional_encoding(idx, context_window) for idx in range(context_window)])


# x: (batch_size, sequence_length, embedding_dimension)
class Embedding(nn.Module):
    def __init__(self, vocab_size: int = 1024):
        super().__init__()
        self.linear1 = nn.Linear(in_features=vocab_size, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)

    def forward(self, x):
        return self.linear2(self.linear1(x)) + positional_encoding_matrix


class AttentionHead(nn.Module):
    def __init__(self, d_embedding, debug: bool = False):
        super().__init__()

        self.input_linear = nn.Linear(d_embedding, d_embedding)
        self.output_linear = nn.Linear(d_embedding, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
        self.softmax = nn.Softmax(dim=-1)
        self.debug = debug

    def forward(self, x):
        if self.debug: print(f"**Inside individual head...")
        Q = self.input_linear(x)
        K = self.input_linear(x)
        V = self.input_linear(x)
        if self.debug: print(f"Q Shape:{Q.shape}")
        if self.debug: print(f"K Shape:{K.shape}")
        QK = torch.matmul(Q, torch.transpose(K, 1, 2))
        scaled_QK = torch.div(QK, torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32)))
        if self.debug: print(f"QK Shape: {scaled_QK.shape}")

        QK = self.softmax(
            scaled_QK
        )

        x = self.output_linear(torch.matmul(QK, V))

        if self.debug: print(f"Attn Output Shape: {x.shape}")

        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_embedding, num_heads=4, debug: bool = False):
        super().__init__()
        self.attn_width = d_embedding // num_heads
        self.d_embedding = d_embedding
        self.debug = debug
        if self.debug: print(f"Each head decided to have {self.d_embedding} width of the embedding.")

        [self.add_module(f"attn{idx}", AttentionHead(self.attn_width, debug)) for idx in
         range(num_heads)]
        self.last_linear = PositionwiseFeedForward(d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, x):
        if self.debug: print(f"****Inside multi-head attention...")
        heads = [head for head in self.children() if isinstance(head, AttentionHead)]
        head_outs = [head(x[..., self.attn_width * idx:self.attn_width * (idx + 1)]) for idx, head in enumerate(heads)]
        y = torch.concat(head_outs, dim=2)
        y = y + x
        y = self.layer_norm(y)
        if self.debug: print(f"Concat heads Shape: {y.shape}")
        y_before_linear = y
        y = self.last_linear(y)
        if self.debug: print(f"After linear: {y.shape}")
        y += y_before_linear
        y = self.layer_norm(y)
        return y


class TransformerClassifier(nn.Module):
    def __init__(self, encoder_list: [EncoderBlock], d_sequence: int = 1024, class_num: int = 2):
        super().__init__()
        self.encoders = nn.Sequential(*encoder_list)
        first = encoder_list[0]
        self.classifier = nn.Sequential(
            nn.Linear(first.d_embedding * d_sequence, class_num)
        )

    def forward(self, x):
        y = self.encoders(x)
        y = torch.flatten(y, start_dim=1)
        return self.classifier(y)


# x: (batch_size, sequence_length, embedding_dimension)
class PositionwiseFeedForward(nn.Module):
    # "Implements FFN equation."
    def __init__(self, d_embedding, d_ff=2048, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_embedding, d_ff)
        self.w_2 = nn.Linear(d_ff, d_embedding)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
