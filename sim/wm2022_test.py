import numpy as np
import netsquid as ns
from netsquid.qubits import set_qstate_formalism, QFormalism
from netsquid.qubits import create_qubits, operate, ketstates, StateSampler
from netsquid.qubits import qubitapi
from netsquid.qubits import operators as ops
from netsquid.qubits.ketstates import s0, b00, y0
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.qubitapi import assign_qstate, measure, gmeasure, amplitude_dampen, fidelity

set_qstate_formalism(QFormalism.DM)

# Bell状態の作成
def create_bell_state():
    qA, qB = create_qubits(2)
    assign_qstate(qA, ketstates.s0)  # |0⟩
    assign_qstate(qB, ketstates.s0)  # |0⟩
    operate(qA, ns.H)                # Hadamard on qA
    operate([qA, qB], ns.CX)         # CNOT
    return qA, qB

# テレポーテーション（alice side）
def tele_alice(qin, qe):
    operate([qin, qe], ns.CX)
    operate(qin, ns.H)
    m0 = measure(qin)
    m1 = measure(qe)
    return m0, m1

# テレポーテーション（bob side）
def tele_bob(m0, m1, qe):
    if (m1 == 1):
        operate(qe, ns.X)
    if (m0 == 1):
        operate(qe, ns.Z)
    return qe

# entanglementの用意
qa, qb = create_bell_state()
print(ns.qubits.reduced_dm([qa, qb]))

# 測定演算子(wm)
omega = np.pi / 3   # 測定強度
m0 = [[np.cos(omega/2), 0], [0, np.sin(omega/2)]]
m1 = [[np.sin(omega/2), 0], [0, np.cos(omega/2)]]
M0 = ops.Operator("M0", m0)
M1 = ops.Operator("M1", m1)
meas_ops = [M0, M1]

# 振幅減衰のKraus演算子
r = 0.6   # 減衰率
e0 = [[1, 0], [0, np.sqrt(1-r)]]
e1 = [[0, np.sqrt(r)], [0, 0]]
E0 = ops.Operator("E0", e0)
E1 = ops.Operator("E1", e1)

# 測定演算子(wmr)
theta = 0.3
n0 = [[theta, 0], [0, 1]]
n0_ = [[np.sqrt(1-theta*theta), 0], [0, 0]]
n1 = [[1, 0], [0, theta]]
n1_ = [[0, 0], [0, np.sqrt(1-theta*theta)]]
N0 = ops.Operator("n0", n0)
N0_ = ops.Operator("n0_", n0_)
N1 = ops.Operator("n1", n1)
N1_ = ops.Operator("n1_", n1_)
wmr_ops0 = [N0, N0_]
wmr_ops1 = [N1, N1_]

# 弱測定
mresult = gmeasure(qb, meas_operators=meas_ops)
print(mresult)
#print(ns.qubits.reduced_dm([qa, qb]))
# フリップ操作
if (mresult[0] == 1):
    operate(qb, ns.X)
#print(ns.qubits.reduced_dm([qa, qb]))
# 振幅減衰
amplitude_dampen(qb, gamma=0.8, prob=1)
print(ns.qubits.reduced_dm([qa, qb]))
# ポストフリップ操作
if (mresult[0] == 1):
    operate(qb, ns.X)
# 逆弱測定
if (mresult[0] == 0):
    mrresult = gmeasure(qb, meas_operators=wmr_ops0)
    if (mrresult[0] == 1):
        print("FAIL")
else:
    mrresult = gmeasure(qb, meas_operators=wmr_ops1)
    if (mrresult[0] == 1):
        print("FAIL")
print(ns.qubits.reduced_dm([qa, qb]))

q1, q2 = create_qubits(2)
assign_qstate([q1, q2], b00)
amplitude_dampen(q2, gamma=0.8, prob=1)
print(ns.qubits.reduced_dm([q1, q2]))
print(fidelity([qa, qb], b00))
print(fidelity([q1, q2], b00))

# テレポーテーション
q, = create_qubits(1)
assign_qstate(q, y0)
tresult = tele_alice(q, qa)
qout = tele_bob(tresult[0][0], tresult[1][0], qb)
print(fidelity(qout, y0))

qt, = create_qubits(1)
assign_qstate(qt, y0)
ttresult = tele_alice(qt, q1)
qtout = tele_bob(ttresult[0][0], ttresult[1][0], q2)
print(fidelity(qtout, y0))