import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits import create_qubits, operate
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

class Bennet(NodeProtocol):
    def __init__(self, node, port, role, start_expression=None, msg_header="bennet", name=None):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "BennetNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        self.role = role
        self._rotprog = self._rotate_program()
        self._measprog = self._measure_program()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_positions = [None, None]
        self._waiting_on_second_qubit = False
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _rotate_program(self):
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_Y, [q1])
        prog.apply(INSTR_Y, [q2])
        return prog
    
    def _measure_program(self):
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_CNOT, [q2, q1])
        prog.apply(INSTR_MEASURE, q1, output_key="M", inplace=False)
        return prog

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            expr = yield cchannel_ready | qmemory_ready
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_meas_result = classical_message.items
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self) # エンタングルメントが保存されたメモリポジションを取得
                #print(f"{self.name}: Entanglement received at {ready_signal.result} / time: {sim_time()}")
                yield from self._handle_new_qubit(ready_signal.result)
            self._check_success()
    
    def start(self):
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self): # 失敗した場合、エンタングルメントを破棄
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]

    def _handle_new_qubit(self, memory_position):
        assert not self.node.qmemory.mem_positions[memory_position].is_empty
        # 2つ目のエンタングルメントが到着した場合
        if self._waiting_on_second_qubit:
            assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
            assert memory_position != self._qmem_positions[0]
            self._qmem_positions[1] = memory_position
            self._waiting_on_second_qubit = False
            yield from self._node_do_bennet()
        # 1つ目のエンタングルメントが到着した場合
        else:
            # Pop previous qubit if present:
            pop_positions = [p for p in self._qmem_positions if p is not None and p != memory_position]
            if len(pop_positions) > 0:
                self.node.qmemory.pop(positions=pop_positions)
            # Set new position:
            self._qmem_positions[0] = memory_position
            self._qmem_positions[1] = None
            self.local_qcount += 1
            self.local_meas_result = None
            self._waiting_on_second_qubit = True

    def _node_do_bennet(self):
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        if self.role.upper() == "A":
            yield self.node.qmemory.execute_program(self._rotprog, [pos1, pos2])
        yield self.node.qmemory.execute_program(self._measprog, [pos1, pos2])
        self.local_meas_result = self._measprog.output["M"][0]
        self._qmem_positions[1] = None
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))
        
    def _check_success(self):
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                self.send_signal(Signals.SUCCESS, self._qmem_positions[0])
                #print(f"{self.name}: SUCCESS / time: {sim_time()}")
            else:
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
                #print(f"{self.name}: FAIL / time: {sim_time()}")
            self.local_meas_result = None
            self.remote_meas_result = None
            self._qmem_positions = [None, None]

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 2:
            return False
        return True

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
        
        self.add_subprotocol(Bennet(node_a, node_a.ports["cout_bob"], role="A", name="bennet_A"))
        self.add_subprotocol(Bennet(node_b, node_b.ports["cin_alice"], role="B", name="bennet_B"))

        self.subprotocols["entangle_A"].start_expression = (
            self.subprotocols["entangle_A"].await_signal(self.subprotocols["bennet_A"], Signals.FAIL) |
                             self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING))
                                
        self.subprotocols["bennet_A"].start_expression = (
            self.subprotocols["bennet_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["bennet_B"].start_expression = (
            self.subprotocols["bennet_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
    
    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["bennet_A"], Signals.SUCCESS) &
                    self.await_signal(self.subprotocols["bennet_B"], Signals.SUCCESS))
            signal_A = self.subprotocols["bennet_A"].get_signal_result(Signals.SUCCESS, self)
            signal_B = self.subprotocols["bennet_B"].get_signal_result(Signals.SUCCESS, self)                                                     
            result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, result)

def sim_setup(node_a, node_b, num_runs):
    be_example = BennetExample(node_a, node_b, num_runs=num_runs)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity([q_A, q_B], ks.b11, squared=True)
        print(f2)
        return {"F2": f2, "pairs": result["pairs"], "time": result["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=be_example,
                                     event_type=Signals.SUCCESS.value))
    return be_example, dc


network = network_setup()
be_example, dc = sim_setup(network.get_node("node_A"), network.get_node("node_B"), num_runs=1)
be_example.start()
ns.sim_run()
#print("Average fidelity of generated entanglement with bennet: {}".format(dc.dataframe["F2"].mean()))
