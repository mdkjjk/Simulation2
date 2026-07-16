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

# Werner状態のペアを２つ用意
qA, qB = create_werner_state(fidelity=0.9)
qC, qD = create_werner_state(fidelity=0.9)
print(qubitapi.reduced_dm([qC, qD]))
print(fidelity([qC, qD], ketstates.b11))    # 初期忠実度

# Aliceサイドで、σ_y回転ゲートを各ペアに適用 => |Φ⁺⟩が主成分になる
operate(qA, ns.Y)
operate(qC, ns.Y)

# 各サイドで、CNOTゲートを適用
operate([qC, qA], ns.CX)
operate([qD, qB], ns.CX)
# 各サイドで、ターゲットビットをZ軸で測定
ma = measure(qA, discard=True)
mb = measure(qB, discard=True)

if(ma[0] == mb[0]):  # 測定結果が一致する場合
    operate(qC, ns.Y)                         # Aliceサイドで、σ_y回転ゲートを制御ビットに適用 => 主成分を|Ψ⁻⟩(Werner状態)に戻す
    print(qubitapi.reduced_dm([qC, qD]))
    print(fidelity([qC, qD], ketstates.b11))  # 精製後の忠実度(>初期忠実度)
else:                # 測定結果が不一致の場合
    print("not match")
    # 両サイドの制御ビットを破棄
    discard(qC)
    discard(qD)