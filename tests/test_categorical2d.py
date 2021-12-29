from drlqap.utils import Categorical2D
import torch
import pytest
import numpy as np

def test_simple_valid():
    Categorical2D(logits=torch.tensor([[1.0, -1.0], [-5.0, 8.0]]), shape=(2,2))

def test_fail_wrong_shape():
    with pytest.raises(Exception):
        Categorical2D(logits=torch.tensor([[1.0, -1.0], [-5.0, 8.0]]), shape=(2,3))

def test_fail_invalid_shape1d():
    with pytest.raises(Exception):
        Categorical2D(logits=torch.tensor([[1.0, -1.0], [-5.0, 8.0]]), shape=(2,))

def test_valid_shape_unsqueezed():
    Categorical2D(logits=torch.tensor([[[1.0], [-1.0]], [[-5.0], [8.0]]]), shape=(2,2))

def test_fail_invalid_shape3d():
    with pytest.raises(Exception):
        Categorical2D(logits=torch.tensor([[[1.0], [-1.0]], [[-5.0], [8.0]]]), shape=(2,2,1))

class TestCategorical2DFlat:
    @pytest.fixture
    def categorical2d(self):
        return Categorical2D(logits=torch.tensor([[1.0, -1.0], [-5.0, 12.0]]), shape=(2,2))

    def test_unravel_1_0(self, categorical2d: Categorical2D):
        assert(categorical2d.unravel((1,0) == 2))

    def test_unravel_0_1(self, categorical2d: Categorical2D):
        assert(categorical2d.unravel((0,1) == 1))

    def test_ravel_1_0(self, categorical2d: Categorical2D):
        assert(categorical2d.ravel((1,0)) == 2)

    def test_ravel_0_1(self, categorical2d: Categorical2D):
        assert(categorical2d.ravel((0,1)) == 1)

    def test_sample(self, categorical2d: Categorical2D):
        # Might fail with probabilitiy ~1/100000
        assert(categorical2d.sample() == (1,1))

    def test_log_prob_1_1(self, categorical2d: Categorical2D):
        assert(np.isclose(np.exp(categorical2d.log_prob((1,1))), 0.99998099))

    def test_log_prob_0_1(self, categorical2d: Categorical2D):
        assert(np.isclose(np.exp(categorical2d.log_prob((1,0))), 0.00000004139859))


class TestCategorical2DUnsqueezed:
    @pytest.fixture
    def categorical2d(self):
        return Categorical2D(logits=torch.tensor([[[1.0], [-1.0]], [[-5.0], [12.0]]]), shape=(2,2))

    def test_unravel_1_0(self, categorical2d: Categorical2D):
        assert(categorical2d.unravel((1,0) == 2))

    def test_unravel_0_1(self, categorical2d: Categorical2D):
        assert(categorical2d.unravel((0,1) == 1))

    def test_ravel_1_0(self, categorical2d: Categorical2D):
        assert(categorical2d.ravel((1,0)) == 2)

    def test_ravel_0_1(self, categorical2d: Categorical2D):
        assert(categorical2d.ravel((0,1)) == 1)

    def test_sample(self, categorical2d: Categorical2D):
        # Might fail with probabilitiy ~1/100000
        assert(categorical2d.sample() == (1,1))

    def test_log_prob_1_1(self, categorical2d: Categorical2D):
        assert(np.isclose(np.exp(categorical2d.log_prob((1,1))), 0.99998099))

    def test_log_prob_0_1(self, categorical2d: Categorical2D):
        assert(np.isclose(np.exp(categorical2d.log_prob((1,0))), 0.00000004139859))

