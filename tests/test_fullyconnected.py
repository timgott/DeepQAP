from torch.nn import Linear
from drlqap.nnutils import FullyConnected, FullyConnectedShaped
from torch.nn import ReLU

def test_fully_connected_shape():
    shape = [10, 3, 149, 1, 6]
    net = FullyConnectedShaped(shape, ReLU)
    
    modules = list(net)
    assert(len(modules) == 8)

    assert(isinstance(modules[0], Linear))
    assert(isinstance(modules[2], Linear))
    assert(isinstance(modules[4], Linear))
    assert(isinstance(modules[6], Linear))

    assert(isinstance(modules[1], ReLU))
    assert(isinstance(modules[3], ReLU))
    assert(isinstance(modules[5], ReLU))
    assert(isinstance(modules[7], ReLU))

    assert(modules[0].in_features == 10)
    assert(modules[2].in_features == 3)
    assert(modules[4].in_features == 149)
    assert(modules[6].in_features == 1)

    assert(modules[0].out_features == 3)
    assert(modules[2].out_features == 149)
    assert(modules[4].out_features == 1)
    assert(modules[6].out_features == 6)

def test_fully_connected():
    net = FullyConnected(77, 22, 13, 2, ReLU)
    modules = list(net)
    assert(len(modules) == 6)
    assert(modules[0].in_features == 77)
    assert(modules[2].in_features == 22)
    assert(modules[4].in_features == 22)

    assert(modules[0].out_features == 22)
    assert(modules[2].out_features == 22)
    assert(modules[4].out_features == 13)
