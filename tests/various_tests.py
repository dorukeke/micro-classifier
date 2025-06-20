import torch

from encoder import PositionwiseFeedForward


######################################################################################
#                                     TESTS
######################################################################################
def test_positionwise_feed_forward():
    feed_forward = PositionwiseFeedForward(d_embedding=5, d_ff=3, dropout_rate=0)
    input_embeddings = torch.ones(1, 10, 5)
    ff_outputs = feed_forward(input_embeddings)

    print("\n", ff_outputs, "\n", ff_outputs.shape)


def test_enumeration():
    lst = [1, 2, 3]
    print([(num, idx) for idx, num in enumerate(lst)])


def test_tensor_slicing():
    rand = torch.rand(1, 1, 4)
    print(rand, rand[..., 1], rand[..., 1:3], rand[..., 3])


def test_tensor_flattening():
    rand = torch.rand(1, 1, 4)
    flattened = torch.flatten(rand, start_dim=1)
    print(rand, flattened, rand.shape, flattened.shape)