import netsquid as ns
from netsquid.qubits import create_qubits, operate, ketstates, StateSampler
from netsquid.qubits import qubitapi
from netsquid.qubits.qubitapi import assign_qstate, measure, fidelity, discard
from netsquid.components.models import DepolarNoiseModel
from netsquid.qubits.qformalism import QFormalism

ns.set_qstate_formalism(QFormalism.DM)

# ベル状態の作成
def create_bell_state():
    qA, qB = create_qubits(2)
    assign_qstate(qA, ketstates.s0)  # |0⟩
    assign_qstate(qB, ketstates.s1)  # |1⟩
    operate(qA, ns.H)                # Hadamard on qA
    operate([qA, qB], ns.CX)         # CNOT
    operate(qA, ns.Z)                # To get |Ψ⁻⟩ from |Ψ⁺⟩
    return qA, qB

# Werner状態の作成
def create_werner_state(fidelity):
    qA, qB = create_bell_state()
    p = 4/3 * (1 - fidelity)               # 脱分極が起きる確率
    noise_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
    noise_model.error_operation([qA, qB])  # ノイズをかける => Werner状態になる
    return qA, qB

q1, q2 = ns.qubits.create_qubits(2, no_state=True)
ns.qubits.assign_qstate([q1, q2], ketstates.b00)
print(qubitapi.reduced_dm([q1, q2]))

q3, q4 = ns.qubits.create_qubits(2, no_state=True)
ns.qubits.assign_qstate([q3, q4], ketstates.b11)
print(qubitapi.reduced_dm([q3, q4]))
qubitapi.operate(q2, ns.Z)
qubitapi.operate(q2, ns.X)
print(qubitapi.reduced_dm([q1, q2]))

