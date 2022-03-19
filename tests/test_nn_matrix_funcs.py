import torch
from drlqap import nnutils

def test_cartesian_product_matrix():
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])

    expected = torch.tensor([
        [[1,2,5,6], [1,2,7,8]],
        [[3,4,5,6], [3,4,7,8]],
    ])
    result = nnutils.cartesian_product_matrix(t1, t2)
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
    result = nnutils.concat_bidirectional(t)

    assert(torch.equal(result, expected))

def test_dot_product():
    t = torch.tensor([
        [1, 2, 4],
        [5, 6, 8]
    ])
    t2 = torch.tensor([
        [3, 2, 1],
        [0, 4, 5]
    ])
    expected = torch.tensor([
        [
            1*3 + 2*2 + 4*1,
            1*0 + 2*4 + 4*5,
        ],
        [
            5*3 + 6*2 + 8*1,
            5*0 + 4*6 + 8*5
        ],
    ])
    result = nnutils.dot_product_matrix(t, t2)
    assert(torch.equal(result, expected))

