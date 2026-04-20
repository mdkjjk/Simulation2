import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.qubitapi import fidelity, discard,create_qubits, amplitude_dampen, reduced_dm, operate
from netsquid.qubits.ketstates import s00, b00
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
from netsquid.qubits.dmtools import DenseDMRepr
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, INSTR_Y
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models import DepolarNoiseModel
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.protocols.protocol import Signals
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.examples.entanglenodes import EntangleNodes
from pydynaa import EventExpression

#print("This example module is located at: {}".format(ns.examples.entanglenodes.__file__))
#print("This example module is located at: {}".format(ns.examples.teleportation.__file__))
#print("This example module is located at: {}".format(ns.examples.purify.__file__))
#print("This example module is located at: {}".format(ns.qubits.qubitapi.__file__))

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

set_qstate_formalism(QFormalism.DM)
q1, = create_qubits(1)  # |0>
amplitude_dampen(q1, gamma=0.1, prob=1)
print(reduced_dm(q1))
operate([q1], ops.X)  # -> |1>
amplitude_dampen(q1, gamma=0.1, prob=1)
print(reduced_dm(q1))