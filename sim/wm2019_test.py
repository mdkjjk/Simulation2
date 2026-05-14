import numpy as np
import netsquid as ns
from netsquid.qubits import set_qstate_formalism, QFormalism
from netsquid.qubits import create_qubits, ketstates
from netsquid.qubits import operators as ops
from netsquid.qubits.ketstates import s0, s1, h0, h1, y0, y1
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.qubitapi import assign_qstate, gmeasure, fidelity, multi_operate, operate
set_qstate_formalism(QFormalism.DM)
    
# phase dampingのKraus演算子
p = 0.8   # 減衰率
e1 = [[1, 0],[0, np.sqrt(1-p)]]
e2 = [[0, 0],[0, np.sqrt(p)]]
E1 = ops.Operator("E1", e1)
E2 = ops.Operator("E2", e2)

# qubitの作成
q, = create_qubits(1)
assign_qstate(q, ketstates.h0)  # |+⟩
Rx = ops.create_rotation_op(np.pi/3, (1, 1, 0))
operate(q, Rx)
print(ns.qubits.reduced_dm(q))
state = np.array([[ns.qubits.reduced_dm(q)[0][0], ns.qubits.reduced_dm(q)[0][1]],
                  [ns.qubits.reduced_dm(q)[1][0], ns.qubits.reduced_dm(q)[1][1]]])
print(state)

# noiseが作用した後のqubitの状態
multi_operate(q, [E1, E2], weights=(1, 1))
qubit = ns.qubits.reduced_dm(q)
x = np.abs(2 * qubit[0][1].real)
y = np.abs(2 * qubit[0][1].imag)
z = np.abs(2 * qubit[0][0] - 1)
dm = [x, y, z]
print(ns.qubits.reduced_dm(q))
print(dm)
print(fidelity(q, state))

# 測定演算子
theta = 0.3   # 測定強度
mxp = ops.Operator("Mx+", np.cos(theta/2) * outerprod(h0) + np.sin(theta/2) * outerprod(h1))
mxm = ops.Operator("Mx-", np.cos(theta/2) * outerprod(h1) + np.sin(theta/2) * outerprod(h0))
myp = ops.Operator("My+", np.cos(theta/2) * outerprod(y0) + np.sin(theta/2) * outerprod(y1))
mym = ops.Operator("My-", np.cos(theta/2) * outerprod(y1) + np.sin(theta/2) * outerprod(y0))
mzp = ops.Operator("Mz+", np.cos(theta/2) * outerprod(s0) + np.sin(theta/2) * outerprod(s1))
mzm = ops.Operator("Mz-", np.cos(theta/2) * outerprod(s1) + np.sin(theta/2) * outerprod(s0))
meas_ops_x = [mxp, mxm]
meas_ops_y = [myp, mym]
meas_ops_z = [mzp, mzm]

# 回転演算
eta = 0.6   # 補正角度
rxp = [[np.cos(eta/2), -np.sin(eta/2)*1j],[np.sin(eta/2)*1j, np.cos(eta/2)]]
rxm = [[np.cos(eta/2), np.sin(eta/2)*1j],[-np.sin(eta/2)*1j, np.cos(eta/2)]]
ryp = [[np.cos(eta/2), -np.sin(eta/2)],[np.sin(eta/2), np.cos(eta/2)]]
rym = [[np.cos(eta/2), np.sin(eta/2)],[-np.sin(eta/2), np.cos(eta/2)]]
rzp = [[np.exp((eta/2)*1j), 0],[0, np.exp(-(eta/2)*1j)]]
rzm = [[np.exp(-(eta/2)*1j), 0],[0, np.exp((eta/2)*1j)]]

Rxp = ops.Operator("Rx+", rxp)
Rxm = ops.Operator("Rx-", rxm)
Ryp = ops.Operator("Ry+", ryp)
Rym = ops.Operator("Ry-", rym)
Rzp = ops.Operator("Rz+", rzp)
Rzm = ops.Operator("Rz-", rzm)

# 弱測定
axis = dm.index(max(dm))
if axis == 0:
    if dm[1] > dm[2]:
        meas_operators = meas_ops_y
        rot_axis = [Rzp, Rzm]
    else:
        meas_operators = meas_ops_z
        rot_axis = [Ryp, Rym]
elif axis == 1:
    if dm[0] > dm[2]:
        meas_operators = meas_ops_x
        rot_axis = [Rzp, Rzm]
    else:
        meas_operators = meas_ops_z
        rot_axis = [Rxp, Rxm]
else:
    if dm[0] > dm[1]:
        meas_operators = meas_ops_x
        rot_axis = [Ryp, Rym]
    else:
        meas_operators = meas_ops_y
        rot_axis = [Rxp, Rxm]

mresult = gmeasure(q, meas_operators=meas_operators)
print(mresult)

# 回転制御
if (mresult[0] == 0):
    operate(q, rot_axis[0])
else:
    operate(q, rot_axis[1])

print(ns.qubits.reduced_dm(q))   # 保護処理後のqubitの状態
print(fidelity(q, state))   # 忠実度
