import torch
from drlqap.qap import QAP

test_str1 = """
3

1 2 4
4 3 1
5 0 2

0 6 3
2 4 1
8 8 1
"""

test_str2 = """
3

0 2 4
4 0 1
5 0 0

0 6 3
2 0 1
8 8 0
"""

def test_load():
    qap = QAP.from_qaplib_string(test_str1)
    assert torch.equal(qap.A, torch.tensor([
        [0, 2, 4],
        [4, 0, 1],
        [5, 0, 0],
    ], dtype=torch.float32))
    assert torch.equal(qap.B, torch.tensor([
        [0,6,3],
        [2,0,1],
        [8,8,0],
    ], dtype=torch.float32))
    # C_ij = A_ii * B_jj
    assert torch.equal(qap.linear_costs, torch.tensor([
        [0, 4, 1],
        [0, 12, 3],
        [0, 8, 2],
    ], dtype=torch.float32))
    assert qap.fixed_cost == 0

def test_load_no_diagonal():
    qap = QAP.from_qaplib_string(test_str2)
    assert torch.equal(qap.A, torch.tensor([
        [0, 2, 4],
        [4, 0, 1],
        [5, 0, 0],
    ], dtype=torch.float32))
    assert torch.equal(qap.B, torch.tensor([
        [0,6,3],
        [2,0,1],
        [8,8,0],
    ], dtype=torch.float32))
    # C_ij = A_ii * B_jj
    assert torch.equal(qap.linear_costs, torch.zeros(3,3))
    assert qap.fixed_cost == 0

def test_value():
    qap = QAP.from_qaplib_string(test_str1)
    value = qap.compute_value([2,0,1])
    assert value == 80 #1*1+3*0+2*4+4*3+2*8+4*8+5*1+0*2+1*6

def test_value_no_diagonal():
    qap = QAP.from_qaplib_string(test_str2)
    value = qap.compute_value([2,0,1])
    assert value == 71 #4*3+2*8+4*8+5*1+0*2+1*6

def test_subproblem_value():
    qap = QAP.from_qaplib_string(test_str1)
    assert qap.create_subproblem_for_assignment(0,2).compute_value([0,1]) == 80
    assert qap.create_subproblem_for_assignment(1,0).compute_value([1,0]) == 80
    assert qap.create_subproblem_for_assignment(2,1).compute_value([1,0]) == 80

def test_sub_subproblem_value():
    qap = QAP.from_qaplib_string(test_str1)
    subproblem = qap.create_subproblem_for_assignment(1,0)
    assert subproblem.compute_value([1,0]) == 80
    subproblem = subproblem.create_subproblem_for_assignment(0,1)
    assert subproblem.compute_value([0]) == 80
    subproblem = subproblem.create_subproblem_for_assignment(0,0)
    assert subproblem.compute_value([]) == 80

def test_sub_subproblem_small():
    qap = QAP(
        A=torch.tensor([[0.,4.], [5., 0.]]),
        B=torch.tensor([[0.,1.],[8.,0.]]),
        fixed_cost=0,
        linear_costs=torch.tensor([[32.,29.],[14.,5.]])
    )
    subproblem = qap.create_subproblem_for_assignment(0,1)
    assert torch.equal(subproblem.A, torch.tensor([[0.0]]))
    assert torch.equal(subproblem.B, torch.tensor([[0.0]]))
    assert subproblem.fixed_cost == 29.0
    assert torch.equal(subproblem.linear_costs, torch.tensor([[4.*8.+5.*1.+14.]]))
    assert subproblem.compute_value([0]) == 80

def test_sub_subproblem_small_no_diagonal():
    qap = QAP(
        A=torch.tensor([[0.,4.], [5.,0.]]),
        B=torch.tensor([[0.,1.], [8.,0.]]),
        fixed_cost=0,
        linear_costs=torch.zeros(2,2)
    )
    subproblem = qap.create_subproblem_for_assignment(0,1)
    assert subproblem.fixed_cost == 0.0
    assert torch.equal(subproblem.linear_costs, torch.tensor([[4.*8.+5.*1.]]))

def test_subproblem_fixed_cost():
    qap = QAP.from_qaplib_string(test_str1)
    assert qap.fixed_cost == 0
    assert qap.create_subproblem_for_assignment(0, 2).fixed_cost == 1
    assert qap.create_subproblem_for_assignment(1, 0).fixed_cost == 0
    assert qap.create_subproblem_for_assignment(2, 1).fixed_cost == 8

def test_subproblem_linear_cost():
    qap = QAP.from_qaplib_string(test_str1)
    # 32 29
    # 14  5
    assert torch.equal(qap.create_subproblem_for_assignment(1, 0).linear_costs, torch.tensor([
        [qap.A[1,0]*qap.B[0,1] + qap.A[0,1]*qap.B[1,0] + qap.linear_costs[0,1], qap.A[1,0]*qap.B[0,2] + qap.A[0,1]*qap.B[2,0] + qap.linear_costs[0,2]],
        [qap.A[1,2]*qap.B[0,1] + qap.A[2,1]*qap.B[1,0] + qap.linear_costs[2,1], qap.A[1,2]*qap.B[0,2] + qap.A[2,1]*qap.B[2,0] + qap.linear_costs[2,2]]
    ], dtype=torch.float32))

def test_subproblem_linear_cost_no_diagonal():
    qap = QAP.from_qaplib_string(test_str2)
    # 28 28
    #  6  3
    assert torch.equal(qap.create_subproblem_for_assignment(1, 0).linear_costs, torch.tensor([
        [qap.A[1,0]*qap.B[0,1] + qap.A[0,1]*qap.B[1,0], qap.A[1,0]*qap.B[0,2] + qap.A[0,1]*qap.B[2,0]],
        [qap.A[1,2]*qap.B[0,1] + qap.A[2,1]*qap.B[1,0], qap.A[1,2]*qap.B[0,2] + qap.A[2,1]*qap.B[2,0]]
    ], dtype=torch.float32))

def test_subproblem_quadratic_cost():
    qap = QAP.from_qaplib_string(test_str2)
    subproblem = qap.create_subproblem_for_assignment(1,0)
    assert torch.equal(subproblem.A, torch.tensor([[0.,4.], [5., 0.]]))
    assert torch.equal(subproblem.B, torch.tensor([[0.,1.], [8., 0.]]))
