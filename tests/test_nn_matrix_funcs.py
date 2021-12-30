import torch
from drlqap import nn

def test_cartesian_product_matrix():
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])

    expected = torch.tensor([
        [[1,2,5,6], [1,2,7,8]],
        [[3,4,5,6], [3,4,7,8]],
    ])
    result = nn.cartesian_product_matrix(t1, t2)
    assert(torch.equal(result, expected))

def test_concat_bidirectional():
    t = torch.tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    expected = torch.tensor([
        [[1,2,1,2], [3,4,5,6]],
        [[5,6,3,4], [7,8,7,8]],
    ])
    result = nn.concat_bidirectional(t)

    assert(torch.equal(result, expected))
