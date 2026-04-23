import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
import math
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
from netsquid.util.constrainedmap import ValueConstraint
from netsquid.nodes.connections import Connection
from netsquid.components import QuantumMemory

#print("This example module is located at: {}".format(ns.examples.entanglenodes.__file__))
#print("This example module is located at: {}".format(ns.examples.teleportation.__file__))
#print("This example module is located at: {}".format(ns.examples.purify.__file__))
#print("This example module is located at: {}".format(ns.qubits.qubitapi.__file__))

class ClassicalConnection(Connection):
    def __init__(self, length):
        super().__init__(name="ClassicalConnection")
        self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length,
            models={"delay_model": FibreDelayModel()}))
        self.ports['A'].forward_input(
            self.subcomponents["Channel_A2B"].ports['send'])
        self.subcomponents["Channel_A2B"].ports['recv'].forward_output(
            self.ports['B'])

class EntanglingConnection(Connection):
    def __init__(self, length, source_frequency):
        super().__init__(name="EntanglingConnection")
        timing_model = FixedDelayModel(delay=(1e9 / source_frequency))
        qsource = QSource("qsource", StateSampler([ks.b11], [1.0]), num_ports=2,
                          timing_model=timing_model,
                          status=SourceStatus.INTERNAL)
        self.add_subcomponent(qsource)
        damp_rate = 1000
        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2,
                                      models={"quantum_noise_model": AmplitudeNoiseModel(damp_rate),
                                              "delay_model": FibreDelayModel()})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2,
                                      models={"quantum_noise_model": AmplitudeNoiseModel(damp_rate),
                                              "delay_model": FibreDelayModel()})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])

def example_network_setup(node_distance=4e-3, depolar_rate=1e7):
    # Setup nodes Alice and Bob with quantum memories:
    noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    alice = Node(
        "Alice", port_names=['qin_charlie', 'cout_bob'],
        qmemory=QuantumMemory("AliceMemory", num_positions=2))
    alice.ports['qin_charlie'].forward_input(alice.qmemory.ports['qin1'])
    bob = Node(
        "Bob", port_names=['qin_charlie', 'cin_alice'],
        qmemory=QuantumMemory("BobMemory", num_positions=1))
    bob.ports['qin_charlie'].forward_input(bob.qmemory.ports['qin0'])
    # Setup classical connection between nodes:
    c_conn = ClassicalConnection(length=node_distance)
    alice.ports['cout_bob'].connect(c_conn.ports['A'])
    bob.ports['cin_alice'].connect(c_conn.ports['B'])
    # Setup entangling connection between nodes:
    q_conn = EntanglingConnection(length=node_distance, source_frequency=2e7)
    alice.ports['qin_charlie'].connect(q_conn.ports['A'])
    bob.ports['qin_charlie'].connect(q_conn.ports['B'])
    return alice, bob, q_conn, c_conn

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

ns.set_qstate_formalism(ns.QFormalism.DM)
alice, bob, *_ = example_network_setup()
stats = ns.sim_run(15)
qA, = alice.qmemory.peek(positions=[1])
qB, = bob.qmemory.peek(positions=[0])
qA, qB
fidelity = ns.qubits.fidelity([qA, qB], ns.b11)
print(f"Entangled fidelity (after 5 ns wait) = {fidelity:.3f}")