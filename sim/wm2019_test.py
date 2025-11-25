import numpy as np
import netsquid as ns
from netsquid.qubits import create_qubits, operate, ketstates, StateSampler
from netsquid.qubits import qubitapi
from netsquid.qubits import operators as ops
from netsquid.qubits.ketstates import s0, s1, h0, h1, y0, y1
from netsquid.qubits.qubitapi import assign_qstate, measure, fidelity, discard
from netsquid.components.models import DepolarNoiseModel
from netsquid.qubits.qformalism import QFormalism

# 共役転置行列（随伴行列）を求める関数
def hermitian(arr):
    return np.conjugate(arr.T)

# phase dampingのKraus演算子
p = 0.3   # 減衰率
e1 = [[1, 0],[0, np.sqrt(1-p)]]
e2 = [[0, 0],[0, np.sqrt(p)]]
E1 = ops.Operator("E1", e1)
E2 = ops.Operator("E2", e2)
E1t = hermitian(E1.arr)
E2t = hermitian(E2.arr)
print(E1.arr)
print(E2.arr)

# qubitの作成
q = create_qubits(1)
assign_qstate(q, ketstates.h0)  # |+⟩
print(ns.qubits.reduced_dm(q))

# noiseが作用した後のqubitの状態
qd = E1.arr @ ns.qubits.reduced_dm(q) @ E1t + E2.arr @ ns.qubits.reduced_dm(q) @ E2t
print(qd)

# 測定演算子
theta = 0.8
mxp = ops.Operator("Mx+", np.cos(theta/2) * outerprod(h0) + np.sin(theta/2) * outerprod(h1))
mxm = ops.Operator("Mx-", np.cos(theta/2) * outerprod(h1) + np.sin(theta/2) * outerprod(h0))
myp = ops.Operator("My+", np.cos(theta/2) * outerprod(y0) + np.sin(theta/2) * outerprod(y1))
mym = ops.Operator("My-", np.cos(theta/2) * outerprod(y1) + np.sin(theta/2) * outerprod(y0))
mzp = ops.Operator("Mz+", np.cos(theta/2) * outerprod(s0) + np.sin(theta/2) * outerprod(s1))
mzm = ops.Operator("Mz-", np.cos(theta/2) * outerprod(s1) + np.sin(theta/2) * outerprod(s0))

# 回転演算
