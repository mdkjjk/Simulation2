import numpy as np
import netsquid as ns
import math

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.qformalism import QFormalism
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.util.constrainedmap import ValueConstraint

ns.set_qstate_formalism(QFormalism.DM)

# 時間依存振幅減衰ノイズ
def delay_amplitude_dampen(qubit, gamma, delay):
    if gamma < 0:
        raise ValueError(f"damp_rate {gamma} should be non-negative.")

    # γ = 1 - exp(-Rt)
    damp_rate = 1. - math.exp(- delay * gamma * 1e-9)

    # 振幅減衰を適用
    qapi.amplitude_dampen(qubit, damp_rate)

    return damp_rate

# 振幅減衰ノイズモデル
class AmplitudeNoiseModel(QuantumErrorModel):
    def __init__(self, gamma, time_independent=False, **kwargs):
        super().__init__(**kwargs)
        # NOTE time independence should be set *before* the rate
        self.add_property('time_independent', time_independent, value_type=bool)

        def gamma_constraint(value):
            if self.time_independent and not 0 <= value <= 1:
                return False
            elif value < 0:
                return False
            return True
        self.add_property('gamma', gamma,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(gamma_constraint))
    
    @property
    def gamma(self):
        return self.properties['gamma']
    
    @gamma.setter
    def gamma(self, value):
        self.properties['gamma'] = value

    @property
    def time_independent(self):
        """bool: Whether the probability of depolarizing is time independent."""
        return self.properties['time_independent']

    @time_independent.setter
    def time_independent(self, value):
        self.properties['time_independent'] = value
    
    def error_operation(self, qubits, delta_time=0, **kwargs):
        if self.time_independent:   # 時間非依存
            for qubit in qubits:
                if qubit is not None:
                    qapi.amplitude_dampen(qubit, gamma)
        else:                       # 時間依存
            for qubit in qubits:
                if qubit is not None:
                    delay_amplitude_dampen(qubit, gamma=self.gamma, delay=delta_time)

#時間非依存位相減衰ノイズ
def phase_dampen(qubit, prob=1):
    krausops1 = ops.Operator("E1", [[1, 0],[0, np.sqrt(1-prob)]])
    krausops2 = ops.Operator("E2", [[0, 0],[0, np.sqrt(prob)]])
    krausops = [krausops1, krausops2]

    qapi.multi_operate([qubit], krausops)

#時間依存位相減衰ノイズ
def delay_phase_dampen(qubit, gamma, delay):
    if gamma < 0:
        raise ValueError(f"damp_rate {gamma} should be non-negative.")
    
    damp_rate = 1 - np.exp(-2 * gamma * delay * 1e-9)
    phase_dampen(qubit, damp_rate)

#位相減衰ノイズモデル
class PhaseNoiseModel(QuantumErrorModel):
    def __init__(self, gamma, time_independent=False, **kwargs):
        super().__init__(**kwargs)
        # NOTE time independence should be set *before* the rate
        self.add_property('time_independent', time_independent, value_type=bool)

        def gamma_constraint(value):
            if self.time_independent and not 0 <= value <= 1:
                return False
            elif value < 0:
                return False
            return True
        self.add_property('gamma', gamma,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(gamma_constraint))
    
    @property
    def gamma(self):
        return self.properties['gamma']
    
    @gamma.setter
    def gamma(self, value):
        self.properties['gamma'] = value

    @property
    def time_independent(self):
        """bool: Whether the probability of depolarizing is time independent."""
        return self.properties['time_independent']

    @time_independent.setter
    def time_independent(self, value):
        self.properties['time_independent'] = value
    
    def error_operation(self, qubits, delta_time=0, **kwargs):
        if self.time_independent:   # 時間非依存
            for qubit in qubits:
                if qubit is not None:
                    phase_dampen(qubit, self.gamma)
        else:                       # 時間依存
            for qubit in qubits:
                if qubit is not None:
                    delay_phase_dampen(qubit, gamma=self.gamma, delay=delta_time)