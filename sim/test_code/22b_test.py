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
    qa, qb = create_qubits(2)
    assign_qstate(qa, ketstates.s0)  # |0⟩
    assign_qstate(qb, ketstates.s0)  # |0⟩
    operate(qa, ns.H)                # Hadamard on qa
    operate([qa, qb], ns.CX)         # CNOT
    return qa, qb

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

# 測定演算子(wm)
omega = np.pi / 4   # 測定強度
m0 = [[np.cos(omega/2), 0], [0, np.sin(omega/2)]]
m1 = [[np.sin(omega/2), 0], [0, np.cos(omega/2)]]
M0 = ops.Operator("M0", m0)
M1 = ops.Operator("M1", m1)
meas_ops = [M0, M1]

# 測定演算子(wmr)
theta = 0.2
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

# 保護処理
def protect(qa, qb, meas_ops, wmr_ops0, wmr_ops1):
    # 弱測定
    mresult = gmeasure(qb, meas_operators=meas_ops)
    print("weak measurement result =", mresult[0])
    #print(ns.qubits.reduced_dm([qa, qb]))
    # フリップ操作
    if (mresult[0] == 1):
        operate(qb, ns.X)
    #print(ns.qubits.reduced_dm([qa, qb]))
    # 振幅減衰
    amplitude_dampen(qb, gamma=0.4, prob=1)
    #print(ns.qubits.reduced_dm([qa, qb]))
    # ポストフリップ操作
    if (mresult[0] == 1):
        operate(qb, ns.X)
        mrresult = gmeasure(qb, meas_operators=wmr_ops1)
        print("WMR result              =", mrresult[0])
        if (mrresult[0] == 1):
            #print("FAIL")
            result = 0
        else:
            #print("SUCCESS")
            result = 1
            print("after protect:")
            print(ns.qubits.reduced_dm([qa, qb]))
    else:   # 逆弱測定
        mrresult = gmeasure(qb, meas_operators=wmr_ops0)
        print("WMR result              =", mrresult[0])
        if (mrresult[0] == 1):
            #print("FAIL")
            result = 0
        else:
            #print("SUCCESS")
            result = 1
            print("after protect:")
            print(ns.qubits.reduced_dm([qa, qb]))
    return result

# 1組目
pair_1 = 0

while pair_1 != 1:
    qa1, qb1 = create_bell_state()
    pair_1 = protect(
        qa1, qb1,
        meas_ops, wmr_ops0, wmr_ops1
    )

# 2組目
pair_2 = 0

while pair_2 != 1:
    qa2, qb2 = create_bell_state()
    pair_2 = protect(
        qa2, qb2,
        meas_ops, wmr_ops0, wmr_ops1
    )

#print(pair_1, pair_2)

# 精製前
rho1 = ns.qubits.reduced_dm([qa1, qb1])
rho2 = ns.qubits.reduced_dm([qa2, qb2])

print("===== BEFORE PURIFICATION =====")

print("Pair 1:")
print(rho1)
print("F1 =", fidelity([qa1, qb1], ketstates.b00))

print()

print("Pair 2:")
print(rho2)
print("F2 =", fidelity([qa2, qb2], ketstates.b00))

print()

print("rho1 == rho2 :", np.allclose(rho1, rho2))

# BXOR
operate([qa1, qa2], ns.CX)
operate([qb1, qb2], ns.CX)

# ターゲット測定
ma = measure(qa2, discard=True)
print("Alice measurement =", ma[0])

mb = measure(qb2, discard=True)
print("Bob measurement   =", mb[0])

# 精製後
rho_out = ns.qubits.reduced_dm([qa1, qb1])

print()
print("===== AFTER PURIFICATION =====")
print(rho_out)
print("F =", fidelity([qa1, qb1], ketstates.b00))

if ma[0] == mb[0]:
    print("SUCCESS", ma[0], mb[0])
else:
    print("FAIL", ma[0], mb[0])

q1, q2 = create_qubits(2)
assign_qstate([q1, q2], b00)
amplitude_dampen(q2, gamma=0.4, prob=1)
print(ns.qubits.reduced_dm([q1, q2]))
print("Standard F=", fidelity([q1, q2], b00))

q, = create_qubits(1)
assign_qstate(q, y0)
tresult = tele_alice(q, qa1)
qout = tele_bob(tresult[0][0], tresult[1][0], qb1)
print("Teleportation F =", fidelity(qout, y0))

qt, = create_qubits(1)
assign_qstate(qt, y0)
ttresult = tele_alice(qt, q1)
qtout = tele_bob(ttresult[0][0], ttresult[1][0], q2)
print("Standard teleportation F =",fidelity(qtout, y0))