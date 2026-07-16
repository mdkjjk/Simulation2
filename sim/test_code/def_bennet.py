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
    operate(qA, ns.Z)                # To get |Ψ⁻⟩ from |Φ⁺⟩
    return qA, qB

# Werner状態の作成
def create_werner_state(fidelity=0.7):
    qA, qB = create_bell_state()
    p = 4/3 * (1 - fidelity)               # 脱分極が起きる確率
    noise_model = DepolarNoiseModel(depolar_rate=p, time_independent=True)
    noise_model.error_operation([qA, qB])  # ノイズをかける => Werner状態になる
    return qA, qB

# 課題：失敗したときの条件分岐を整理すること
def bennet_epp(qubits, itteration):
    itteration += 1
    if (itteration == 1): # 1回目の場合
        print("Phase {}".format(itteration))
        q_a1, q_b1 = create_werner_state(fidelity=0.8)   # ペアの共有
        q_a2, q_b2 = create_werner_state(fidelity=0.8)
        print("Before: ")
        print(fidelity([q_a2, q_b2], ketstates.b11))     # 初期エンタングルメント忠実度

        # Aliceのみ回転操作
        operate(q_a1, ns.Y)
        operate(q_a2, ns.Y)
        # 両者でCNOTゲート
        operate([q_a2, q_a1], ns.CX)
        operate([q_b2, q_b1], ns.CX)
        # 両者でターゲットビットをz軸で測定
        ma = measure(q_a1, discard=True)
        mb = measure(q_b1, discard=True)

        if (ma[0] == mb[0]): # 測定結果が一致する場合
            # Aliceのみ回転操作
            operate(q_a2, ns.Y)
            print("After: ")
            print(fidelity([q_a2,q_b2], ketstates.b11))  # 精製後のエンタングルメント忠実度
        else: # 測定結果が不一致の場合
            print("FAIL")
            discard(q_a2)   # ペアを破棄
            discard(q_b2)
            qp_a, qp_b = bennet_epp(None, 0)

        return q_a2, q_b2
    else:
        print("Phase {}".format(itteration))
        print("Before: ")
        print(fidelity([qubits[2], qubits[3]], ketstates.b11))
        print(qubitapi.reduced_dm([qubits[2], qubits[3]]))

        operate(qubits[0], ns.Y)
        operate(qubits[2], ns.Y)
        operate([qubits[2], qubits[0]], ns.CX)
        operate([qubits[3], qubits[1]], ns.CX)

        ma = measure(qubits[0], discard=True)
        mb = measure(qubits[1], discard=True)

        if (ma[0] == mb[0]):
            operate(qubits[2], ns.Y)
            print("After: ")
            print(fidelity([qubits[2], qubits[3]], ketstates.b11))
            print(qubitapi.reduced_dm([qubits[2], qubits[3]]))
        else:
            print("FAIL")
            discard(qubits[2])
            discard(qubits[3])
            qp_a, qp_b = bennet_epp(None, 0)

        return qubits[2], qubits[3]

q_1a, q_1b = bennet_epp(None, 0)
q_2a, q_2b = bennet_epp(None, 0)
q_3a, q_3b = bennet_epp([q_1a, q_1b, q_2a, q_2b], 1)