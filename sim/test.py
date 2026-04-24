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

class LocalEntangle(NodeProtocol):
    def __init__(self, node, qsource_name, start_expression=None, 
                 input_mem_pos0=0, input_mem_pos1=1, num_pairs=1, name=None):
        name = name if name else f"LocalEntangle({node.name})"
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._qsource_name = qsource_name
        self._mem_pos0 = input_mem_pos0
        self._mem_pos1 = input_mem_pos1

        # 2つの入力ポート
        self._qin0 = self.node.qmemory.ports[f"qin{self._mem_pos0}"]
        self._qin1 = self.node.qmemory.ports[f"qin{self._mem_pos1}"]

        # メモリ確保
        self.node.qmemory.mem_positions[self._mem_pos0].in_use = True
        self.node.qmemory.mem_positions[self._mem_pos1].in_use = True

    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._mem_pos0, self._mem_pos1]
        # Claim extra memory positions to use (if any):
        extra_memory = self._num_pairs - 1
        if extra_memory > 0:
            unused_positions = self.node.qmemory.unused_positions
            if extra_memory > len(unused_positions):
                raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
                                   .format(self._num_pairs - 1, len(unused_positions)))
            for i in unused_positions[:extra_memory]:
                self._mem_positions.append(i)
                self.node.qmemory.mem_positions[i].in_use = True
        # Call parent start method
        return super().start()

    def stop(self):
        # Unclaim used memory positions:
        if self._mem_positions:
            for i in self._mem_positions[1:]:
                self.node.qmemory.mem_positions[i].in_use = False
            self._mem_positions = None
        # Call parent stop method
        super().stop()

    def run(self):
        while True:
            if self.start_expression is not None:
                yield self.start_expression
            elif self.entangled_pairs >= self._num_pairs:
                break
            self.node.subcomponents[self._qsource_name].trigger()
            yield self.await_port_input(self._qin0)
            yield self.await_port_input(self._qin1)
            self.entangled_pairs += 1
            self.send_signal(Signals.SUCCESS, mem_pos)

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
            return False
        if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            return False
        return True

def network_setup(source_delay=1e5, source_fidelity_sq=0.9, damp_rate=100, node_distance=1000):
    network = Network("wmeasure_network")

    # ノード設定
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=11,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    state_sampler = StateSampler([ks.b00, ks.s00], probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    source_frequency = 4e4 / node_distance
    node_a.add_subcomponent(QSource("QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor("QuantumMemory_B", num_positions=11,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能

    # チャネル設定
    conn_cchannel = DirectConnection("CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel,
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    # quantum_noise_modelに振幅減衰ノイズを指定
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False),
                                      "delay_model": FibreDelayModel(c=200e3)})
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum")
    
    # Link Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].connect(
        node_a.qmemory.ports["qin1"])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    node_a.qmemory.ports["qout"].forward_output(node_a.ports[port_name_a])
    # Link Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    return network

class Example(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="WMeasure example")
        self.num_runs = num_runs
        # エンタングルメント生成プロトコル
        self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A", input_mem_pos0=0,
                                           input_mem_pos1=1, num_pairs=1, name="entangle_A"))

        self.subprotocols["entangle_A"].start_expression = (
                             self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING))

    def run(self):
        self.start_subprotocols()
        start_time = sim_time()
        self.send_signal(Signals.WAITING)
        yield self.await_signal(self.subprotocols["entangle_A"], Signals.SUCCESS)
        self.send_signal(Signals.SUCCESS)

def sim_setup(node_a, node_b, num_runs):
    wm_example = Example(node_a, node_b, num_runs=num_runs)
    return wm_example

network = network_setup()
wm_example = sim_setup(network.get_node("node_A"), network.get_node("node_B"), 1)
wm_example.start()
ns.sim_run()