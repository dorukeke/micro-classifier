import torch

from encoder import (
    TransformerClassifier,
    EncoderBlock
)
from tokenizer import Tokenizer, encode_tokens_one_hot_encoding
from trainer import CustomTrainer


def convert_to_binary_classes(positive):
    if positive:
        return torch.tensor([1, 0], dtype=torch.float32)
    else:
        return torch.tensor([0, 1], dtype=torch.float32)


def convert_from_binary_classes(result):
    high_ind = torch.argmax(result, dim=1)
    if high_ind == 1:
        return 0
    else:
        return 1


tokenizer = Tokenizer()


def text_to_tensor(text):
    return encode_tokens_one_hot_encoding(tokenizer.tokenize(text))


def convert_examples_to_training_batches(_examples: list, batches_of: int = 10) -> ([torch.Tensor], [torch.Tensor]):
    return (
        torch.stack(
            [text_to_tensor(query) for query, result in _examples]
        ).split(batches_of, dim=0),
        torch.stack(
            [convert_to_binary_classes(result) for query, result in _examples]
        ).split(batches_of, dim=0)
    )


device = "cpu"
if torch.cuda.is_available():
    print("CUDA device selected")
    device = torch.cuda.current_device()

POSITIVE = 1
NEGATIVE = 0

examples = [
    ["I am very good", POSITIVE],
    ["I feel terrible", NEGATIVE],
    ["Its a bad day", NEGATIVE],
    ["Its a good day", POSITIVE],
    ["I miss my old days", NEGATIVE],
    ["I am very happy", POSITIVE],
    ["I am very bad", NEGATIVE],
]

debug = False

training_input_batches, training_output_batches = convert_examples_to_training_batches(examples)
if debug: print(f"Input shape: {training_input_batches[0].shape}")
if debug: print(f"Output shape: {training_output_batches[0].shape}")

network = TransformerClassifier(
    encoder_list=[EncoderBlock(d_embedding=1024, num_heads=8, debug=debug)],
    class_num=2
)

CustomTrainer(network, device).train(
    number_of_epochs=1000,
    training_input_batches=training_input_batches,
    training_output_batches=training_output_batches
)


def convert_text_to_inference_batch(text):
    test_data = text_to_tensor(text).to(device)
    return test_data.reshape(1, test_data.shape[0], test_data.shape[1])


def infer_text(input_text):
    result = network(convert_text_to_inference_batch(input_text))
    return f"POSITIVE {result}" if convert_from_binary_classes(result) else f"NEGATIVE {result}"


while True:
    query = input("Enter a text or write 'exit' to stop classifying: ")
    if query == "exit":
        break
    else:
        print(f"{query} is: {infer_text(query)}")
