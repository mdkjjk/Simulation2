import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits import create_qubits, operate, ketstates, StateSampler
from netsquid.qubits.qubitapi import assign_qstate, measure, fidelity, discard
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1, s00, b11
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
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
from netsquid.protocols.protocol import Signals
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.examples.entanglenodes import EntangleNodes
from pydynaa import EventExpression

ns.set_qstate_formalism(QFormalism.DM)

def network_setup(source_delay=1e5, source_fidelity_sq=0.8, fidelity=0.7, node_distance=1000):
    network = Network("bennet_network")

    # ノード設定
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=11,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    #state = DenseDMRepr(np.array[[(1-f)/3, 0, 0, 0], [0, (2*f+1)/6, (1-4*f)/6, 0],
                                 #[0, (1-4*f)/6, (2*f+1)/6, 0], [0, 0, 0, (1-f)/3]])   werner状態の密度行列
    state_sampler = StateSampler([ks.b11, ks.s00], probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
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
    p = 4/3 * (1 - fidelity)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate=p, time_independent=True),
                                      "delay_model": FibreDelayModel(c=200e3)})
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum")
    
    # Link Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    # Link Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    return network

class BennetExample(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Bennet example")
        self.num_runs = num_runs

        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=2, name="entangle_A"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", input_mem_pos=0,
                                           num_pairs=2, name="entangle_B"))

        self.subprotocols["entangle_A"].start_expression = (self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
    
    def run(self):
        self.start_subprotocols()
        self.subprotocols["entangle_A"].entangled_pairs = 0
        self.send_signal(Signals.WAITING)

def sim_setup(node_a, node_b, num_runs):
    be_example = BennetExample(node_a, node_b, num_runs=num_runs)
    return be_example

class RotateProgram(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_Y, q1)
        self.apply(instr.INSTR_Y, q2)
        yield self.run()

class MeasurementProgram(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_CNOT, [q2, q1])
        self.apply(instr.INSTR_MEASURE, q1, output_key="M")
        yield self.run()

class MeasurementProtocol(NodeProtocol):
    def run(self):
        rotate_program = RotateProgram()
        measure_program = MeasurementProgram()
        if self.node.name = "node_A":
            self.node.qmemory.execute_program(rotate_program)
        self.node.qmemory.execute_program(measure_program)
        m, = measure_program.output["M"][0]


network = network_setup()
node_a = network.get_node("node_A")
node_b = network.get_node("node_B")
be_example = sim_setup(node_a, node_b, num_runs=1)
be_example.start()
ns.sim_run(1e10)
qa, = node_a.qmemory.peek(positions=[0])
qb, = node_b.qmemory.peek(positions=[0])
qc, = node_a.qmemory.peek(positions=[1])
qd, = node_b.qmemory.peek(positions=[1])
print(qapi.reduced_dm([qa, qb]))
print(fidelity([qa, qb], ketstates.b11))

