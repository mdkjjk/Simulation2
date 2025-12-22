import numpy as np
import netsquid as ns
from netsquid.qubits import set_qstate_formalism, QFormalism
from netsquid.qubits import create_qubits, operate, ketstates, StateSampler
from netsquid.qubits import qubitapi
from netsquid.qubits import operators as ops
from netsquid.qubits.ketstates import s0, b00, y0, h0
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.qubitapi import assign_qstate, measure, gmeasure, multi_operate, amplitude_dampen, fidelity
from netsquid.qubits.qformalism import QFormalism
from netsquid.components.models import T1T2NoiseModel


set_qstate_formalism(QFormalism.DM)
q1, = create_qubits(1)
gamma = 0.3
prob = 0.5
a = np.sqrt(gamma)
b = np.sqrt(1-gamma)
E0 = ops.Operator("E0_AD", [[1, 0], [0, b]])
E1 = ops.Operator("E1_AD", [[0, a], [0, 0]])
E2 = ops.Operator("E2_AD", [[b, 0], [0, 1]])
E3 = ops.Operator("E3_AD", [[0, 0], [a, 0]])
multi_operate(q1, [E0, E1, E2, E3],
              weights=(prob, prob, 1 - prob, 1 - prob))
print(ns.qubits.reduced_dm(q1))

# phase dampingのKraus演算子
p = 0.8   # 減衰率
e1 = [[1, 0],[0, np.sqrt(1-p)]]
e2 = [[0, 0],[0, np.sqrt(p)]]
E1 = ops.Operator("E1", e1)
E2 = ops.Operator("E2", e2)

# qubitの作成
q, = create_qubits(1)
assign_qstate(q, ketstates.h0)  # |+⟩
print(ns.qubits.reduced_dm(q))

# noiseが作用した後のqubitの状態
multi_operate(q, [E1, E2], weights=(1, 1))
print(ns.qubits.reduced_dm(q))
print(fidelity(q, h0))