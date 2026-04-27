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
from netsquid.qubits.qubitapi import fidelity, discard
from netsquid.qubits.ketstates import s00, b00
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
from netsquid.qubits.dmtools import DenseDMRepr
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, INSTR_X
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
from netsquid.util.constrainedmap import ValueConstraint
from netsquid.examples.entanglenodes import EntangleNodes
from pydynaa import EventExpression

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
            yield (self.await_port_input(self._qin0) | self.await_port_input(self._qin1))
            self.entangled_pairs += 1
            result = {"mem_pos0": self._mem_pos0,
                      "mem_pos1": self._mem_pos1,}
            self.send_signal(Signals.SUCCESS, result)

class WMeasure(NodeProtocol):   # Alice側のプロトコル
    def __init__(self, node, port, start_expression=None, msg_header="wmeasure", omega=np.pi/3, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "WMeasureNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_OK = False
        self.header = msg_header
        self._qmem_positions = [None, None]
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_wmeasurement_operators(omega)

    def _set_wmeasurement_operators(self, omega):
        m0 = ops.Operator("M0", [[np.cos(omega/2), 0], [0, np.sin(omega/2)]])
        m1 = ops.Operator("M1", [[np.sin(omega/2), 0], [0, np.cos(omega/2)]])
        self.wmeas_ops = [m0, m1]

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            expr = yield cchannel_ready | qmemory_ready
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_OK = classical_message.items
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                        event=expr.second_term.triggered_events[0], receiver=self) # エンタングルメントが保存されたメモリポジションを取得
                print(f"{self.name}: Entanglement received at {ready_signal.result} / time: {sim_time()}")
                self._qmem_positions[0] = ready_signal.result["mem_pos0"]
                self._qmem_positions[1] = ready_signal.result["mem_pos1"]
                print(self._qmem_positions)
                yield from self._handle_qubit_rx()

    def start(self):
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_OK = False
        return super().start()

    def stop(self):
        super().stop()
        # TODO should stop clear qmem_pos?
        if self._qmem_positions:
            for i in self._qmem_positions:
                if self.node.qmemory.mem_positions[i].in_use:
                    self.node.qmemory.pop(positions=[i])

    def _handle_qubit_rx(self):
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [pos2], 
                                                       meas_operators=self.wmeas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        self.local_meas_result = output["instr"][0]
        print(f"Result: {self.local_meas_result}")
        self.local_qcount += 1
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result], header=self.header))
        if self.local_meas_result == "1":
            yield self.node.qmemory.execute_instruction(INSTR_X, [pos2])
        print(f"{self.name}")
        self.node.qmemory.pop(positions=pos2)
        self._qmem_positions[1] = None
        self.send_signal(Signals.SUCCESS, self._qmem_positions)
        print("WMeasure Done")
        #self._check_success()

    def _handle_cchannel_rx(self):
        if (self._qmem_positions is not None and
                self.node.qmemory.mem_positions[self._qmem_positions[0]].in_use):
            self._check_success()

    def _check_success(self):
        if not self.remote_meas_OK:
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)
            self.local_meas_result = None
            self.remote_meas_OK = False

    def _handle_fail(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]
        

class RWMeasure(NodeProtocol):   # Bob側のプロトコル
    def __init__(self, node, port_c, port_q, start_expression=None, msg_header="wmeasure", theta=0.2, name=None):
        if not isinstance(port_c, Port) or not isinstance(port_q, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "RWMeasureNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port_c = port_c
        self.port_q = port_q
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_pos = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_rwmeasurement_operators(theta)

    def _set_rwmeasurement_operators(self, theta):
        n0 = ops.Operator("N0", [[theta, 0], [0, 1]])
        n0_ = ops.Operator("N0_", [[np.sqrt(1-theta*theta), 0], [0, 0]])
        n1 = ops.Operator("N1", [[1, 0], [0, theta]])
        n1_ = ops.Operator("N1_", [[0, 0], [0, np.sqrt(1-theta*theta)]])
        self.rwmeas_ops0 = [n0, n0_]
        self.rwmeas_ops1 = [n1, n1_]
    
    def run(self):
        cchannel_ready = self.await_port_input(self.port_c)
        qmemory_ready = self.await_port_input(self.port_q)
        while True:
            yield cchannel_ready
            classical_message = self.port_c.rx_input(header=self.header)
            if classical_message:
                self.remote_qcount, self.remote_meas_result = classical_message.items
                print(f"{self.name}: Alice's result received {classical_message}")
            yield qmemory_ready
            if self.node.qmemory.mem_positions[0].in_use:
                print("Entanglement arrived")
                self._qmem_pos = 0
                yield from self._handle_qubit_rx()
    
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_result = None
        self.remote_meas_result = None
        return super().start()

    def stop(self):
        super().stop()
        # TODO should stop clear qmem_pos?
        if self._qmem_pos and self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    def _handle_qubit_rx(self):
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        if self.remote_meas_result == 1:
            yield self.node.qmemory.execute_instruction(INSTR_X, [pos2])

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
    network.add_connection(node_a, node_b, channel_to=qchannel, label="quantum",
                           port_name_node1="qout_bob", port_name_node2="qin_alice")
    
    # Link Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].connect(
        node_a.qmemory.ports["qin1"])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    node_a.qmemory.ports["qout"].forward_output(node_a.ports["qout_bob"])
    # Link Bob ports:
    node_b.ports["qin_alice"].forward_input(node_b.qmemory.ports["qin0"])
    return network

class WMeasureExample(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="WMeasure example")
        self.num_runs = num_runs
        # エンタングルメント生成プロトコル
        self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A", input_mem_pos0=0,
                                           input_mem_pos1=1, num_pairs=1, name="entangle_A"))
        # 保護処理プロトコル
        self.add_subprotocol(WMeasure(node_a, node_a.ports["cout_bob"], omega=np.pi/3, name="wmeasure_A"))
        self.add_subprotocol(RWMeasure(node_b, node_b.ports["cin_alice"],
                             node_b.ports["qin_alice"], theta=0.2, name="rwmeasure_B"))
        # エンタングルメント生成プロトコルの開始条件
        self.subprotocols["entangle_A"].start_expression = (
                             self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING))
        # 精製処理プロトコルの開始条件                        
        self.subprotocols["wmeasure_A"].start_expression = (
            self.subprotocols["wmeasure_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["rwmeasure_B"].start_expression = (
            self.subprotocols["rwmeasure_B"].await_signal(self, Signals.WAITING))
                                                       
    
    def run(self):
        self.start_subprotocols()
        start_time = sim_time()
        self.subprotocols["entangle_A"].entangled_pairs = 0
        self.send_signal(Signals.WAITING)
        yield self.await_signal(self.subprotocols["rwmeasure_B"], Signals.SUCCESS)
        mem = self.subprotocols["rwmeasure_B"].get_signal_result(Signals.SUCCESS, self)
        print(mem)
        self.send_signal(Signals.SUCCESS)

def sim_setup(node_a, node_b, num_runs):
    wm_example = WMeasureExample(node_a, node_b, num_runs=num_runs)
    return wm_example

network = network_setup()
wm_example = sim_setup(network.get_node("node_A"), network.get_node("node_B"), 1)
wm_example.start()
ns.sim_run()