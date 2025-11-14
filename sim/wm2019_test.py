import numpy as np
import netsquid as ns
from netsquid.qubits import create_qubits, operate, ketstates, StateSampler
from netsquid.qubits import qubitapi
from netsquid.qubits import operators as ops
from netsquid.qubits.qubitapi import assign_qstate, measure, fidelity, discard
from netsquid.components.models import DepolarNoiseModel
from netsquid.qubits.qformalism import QFormalism

# 共役転置行列（随伴行列）を求める関数
def hermitian(arr):
    return np.conjugate(arr.T)

# phase dampingのKraus演算子
p = 0.3
e1 = [[1, 0],[0, np.sqrt(1-p)]]
e2 = [[0, 0],[0, np.sqrt(p)]]
E1 = ops.Operator("E1", e1)
E2 = ops.Operator("E2", e2)
print(E1.arr)
print(E2.arr)

# qubitの作成
q = create_qubits(1)
assign_qstate(q, ketstates.h0)  # |+⟩
print(ns.qubits.reduced_dm(q))

# noiseが作用した後のqubitの状態
qd = 