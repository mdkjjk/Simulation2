import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
from noise import AmplitudeNoiseModel, PhaseNoiseModel
from teleportation import InitStateProgram, BellMeasurement, Correction
from wm2022 import LocalEntangle
from wmb import Protect, RWMeasure, QuantumDispatcher
from deutsch import Distil

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.qubitapi import fidelity, discard
from netsquid.qubits.ketstates import s00, b00, y0
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
from netsquid.qubits.dmtools import DenseDMRepr
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, INSTR_X, IGate
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
from pydynaa import EventExpression

ns.set_qstate_formalism(QFormalism.DM)

class Distil(NodeProtocol):
    """Protocol that does local DEJMPS distillation on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    role : "A" or "B"
        Distillation requires that one of the nodes ("B") conjugate its rotation,
        while the other doesn't ("A").
    start_expression : :class:`~pydynaa.EventExpression`
        EventExpression node should wait for before starting distillation.
        The EventExpression should have a protocol as source, this protocol should signal the quantum memory position
        of the qubit.
    msg_header : str, optional
        Value of header meta field used for classical communication.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """
    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        self.role = role
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self._program = self._setup_dejmp_program(conj_rotation)
        # self.INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        self.num_runs = 0
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_positions = [None, None]
        self._waiting_on_second_qubit = False
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _setup_dejmp_program(self, conj_rotation):
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1])
        prog.apply(INSTR_ROT, [q2])
        prog.apply(INSTR_CNOT, [q1, q2])
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False)
        return prog

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            #print(f"{self.name}: Start")
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    print(f"{self.name}: Result received at {classical_message} / time: {sim_time()}")
            elif expr.second_term.value:
                print(f"{self.name}: Bennett purification start / time: {sim_time()}")

                # ----------------------------------------------
                # 使用する量子ビットのメモリ位置を設定
                # ----------------------------------------------
                if self.role.upper() == "A":
                    self._qmem_positions = [0, 2]
                else:
                    self._qmem_positions = [0, 1]
                self.local_qcount += 1
                print(f"{self.name}: Bennett qubits = {self._qmem_positions}")
                yield from self._node_do_DEJMPS()
            self._check_success()

    def start(self):
        # Clear any held qubits
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        #print(f"{self.name}: pop_positions = {positions} at _clear_qmem_positions")
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]
        #print(f"{self.name}: qmem_positions = {self._qmem_positions}")

    def _node_do_DEJMPS(self):
        self.num_runs += 1
        # Perform DEJMPS distillation protocol locally on one node
        #print(f"{self.name}: qmem_positions = {self._qmem_positions}")
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # We perform local DEJMPS
        yield self.node.qmemory.execute_program(self._program, [pos1, pos2])  # If instruction not instant
        self.local_meas_result = self._program.output["m"][0]
        self._qmem_positions[1] = None
        #print(f"{self.name}: qmem_positions = {self._qmem_positions}")
        # Send local results to the remote node to allow it to check for success.
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))

    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, [self._qmem_positions[0], self.num_runs])
                print(f"{self.name}: SUCCESS / time: {sim_time()}")
                self.num_runs = 0
                self.local_qcount = 0
            else:
                # FAILURE
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
                print(f"{self.name}: FAIL / time: {sim_time()}")
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

class ProtectDeutsch(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs, omega, theta):
            super().__init__(nodes={"A": node_a, "B": node_b}, name="Protect&deutsch")
            self.num_runs = num_runs
            # エンタングルメント生成プロトコル
            self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A1", input_mem_pos0=0,
                                               input_mem_pos1=1, num_pairs=1, name="entangle_A1"))
            self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A2", input_mem_pos0=2,
                                               input_mem_pos1=3, num_pairs=1, name="entangle_A2"))
            # 保護処理プロトコル
            self.add_subprotocol(Protect(node_a, node_a.ports["cout_bob1"], omega=omega, pair_id=1, name="protect_A1"))
            self.add_subprotocol(Protect(node_a, node_a.ports["cout_bob2"], omega=omega,pair_id=2, name="protect_A2"))
            self.add_subprotocol(QuantumDispatcher(node_b, node_b.ports["qdispatch_in"], name="quantum_dispatcher"))
            self.add_subprotocol(RWMeasure(node_b, node_b.ports["cin_alice1"], self.subprotocols["quantum_dispatcher"],
                                         pair_id=1, theta=theta, name="rwmeasure_B1"))
            self.add_subprotocol(RWMeasure(node_b, node_b.ports["cin_alice2"], self.subprotocols["quantum_dispatcher"],
                                         pair_id=2, theta=theta, name="rwmeasure_B2"))
            # 精製処理プロトコル
            self.add_subprotocol(Distil(node_a, node_a.ports["cout_bob"], role="A", name="deutsch_A"))
            self.add_subprotocol(Distil(node_b, node_b.ports["cin_alice"], role="B", name="deutsch_B"))
            # テレポーテーションプロトコル
            self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
            self.add_subprotocol(Correction(node=node_b, name="teleport_B"))

            # エンタングルメント生成プロトコルの開始条件
            self.subprotocols["entangle_A1"].start_expression = (
                                    self.subprotocols["entangle_A1"].await_signal(self, Signals.WAITING) |
                                    self.subprotocols["entangle_A1"].await_signal(self.subprotocols["protect_A1"], Signals.FAIL) |
                                    self.subprotocols["entangle_A1"].await_signal(self.subprotocols["deutsch_A"], Signals.FAIL))
            self.subprotocols["entangle_A2"].start_expression = (
                                    self.subprotocols["entangle_A2"].await_signal(self.subprotocols["protect_A1"], Signals.SUCCESS) |
                                    self.subprotocols["entangle_A2"].await_signal(self.subprotocols["protect_A2"], Signals.FAIL))
                                        
            # 保護処理プロトコルの開始条件                        
            self.subprotocols["protect_A1"].start_expression = (
                self.subprotocols["protect_A1"].await_signal(self.subprotocols["entangle_A1"], Signals.SUCCESS))
            self.subprotocols["protect_A2"].start_expression = (
                self.subprotocols["protect_A2"].await_signal(self.subprotocols["entangle_A2"], Signals.SUCCESS))
            self.subprotocols["rwmeasure_B1"].start_expression = (
                self.subprotocols["rwmeasure_B1"].await_signal(self, Signals.WAITING) |
                self.subprotocols["rwmeasure_B1"].await_signal(self.subprotocols["protect_A1"], Signals.FAIL))
            self.subprotocols["rwmeasure_B2"].start_expression = (
                            self.subprotocols["rwmeasure_B2"].await_signal(self.subprotocols["entangle_A1"], Signals.SUCCESS) |
                            self.subprotocols["rwmeasure_B2"].await_signal(self.subprotocols["protect_A2"], Signals.FAIL))

            # 精製処理プロトコルの開始条件                        
            self.subprotocols["deutsch_A"].start_expression = (
                self.subprotocols["deutsch_A"].await_signal(self.subprotocols["protect_A2"], Signals.SUCCESS))
                                                            
            self.subprotocols["deutsch_B"].start_expression = (
                self.subprotocols["deutsch_B"].await_signal(self.subprotocols["rwmeasure_B2"], Signals.SUCCESS))  

            # テレポーテーションプロトコルの開始条件
            self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(
                                                                self.subprotocols["deutsch_A"], Signals.SUCCESS)
            self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(
                                                                self.subprotocols["deutsch_B"], Signals.SUCCESS)
                                                                                                         

    def run(self):
            self.start_subprotocols()
            for i in range(self.num_runs):
                #print(f"Simulation {i}")
                start_time = sim_time()
                self.subprotocols["entangle_A1"].entangled_pairs = 0
                self.subprotocols["entangle_A2"].entangled_pairs = 0
                self.send_signal(Signals.WAITING)
                yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS)
                       & self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS))
                signal_A = self.subprotocols["deutsch_A"].get_signal_result(Signals.SUCCESS, self)
                result_en = {
                    "pairs1": self.subprotocols["entangle_A1"].entangled_pairs,
                    "pairs2": self.subprotocols["entangle_A2"].entangled_pairs,
                    "runs": signal_A[1],
                    "time": sim_time() - start_time
                }
                result_A = self.subprotocols["teleport_A"].get_signal_result(Signals.SUCCESS, self)
                result_B = self.subprotocols["teleport_B"].get_signal_result(Signals.SUCCESS, self)
                result_tel = {
                    "pos_A0": result_A["pos_A0"],
                    "pos_A1": result_A["pos_A1"],
                    "pos_B": result_B,
                    "time": sim_time() - start_time
                }
                self.send_signal(Signals.SUCCESS, [result_en, result_tel])

def sim_setup(node_a, node_b, num_runs, omega, theta):
    pb_example = ProtectDeutsch(node_a, node_b, num_runs, omega, theta)
    
    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result_en, result_tel = protocol.get_signal_result(Signals.SUCCESS)
        print(result_en)
        node_a.qmemory.discard(positions=[result_tel["pos_A0"]]) # 使用しているメモリを解放
        node_a.qmemory.discard(positions=[result_tel["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result_tel["pos_B"]])
        #print(qapi.reduced_dm([q_A, q_B]))
        pairs = result_en["pairs1"] + result_en["pairs2"]
        prob = 1 / result_en["runs"]
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)
        return {"fidelity": f2, "pairs": pairs, "probability": prob, "time": result_en["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                        include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=pb_example,
                                        event_type=Signals.SUCCESS.value))
    return pb_example, dc

def network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=200, node_distance=200):
    network = Network("wmeasure_network")

    # ノード設定
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=6,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    state_sampler = StateSampler([ks.b00, ks.s00], probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    source_frequency = 4e4 / node_distance
    node_a.add_subcomponent(QSource("QSource_A1", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_a.add_subcomponent(QSource("QSource_A2", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor("QuantumMemory_B", num_positions=6,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能

    # チャネル設定
    conn_cchannel1 = DirectConnection("CChannelConn_AB_1",
        ClassicalChannel("CChannel_A1->B1", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B1->A1", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel1, label="channel1",
                           port_name_node1="cout_bob1", port_name_node2="cin_alice1")
    conn_cchanne2 = DirectConnection("CChannelConn_AB_2",
        ClassicalChannel("CChannel_A2->B2", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B2->A2", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchanne2, label="channel2",
                            port_name_node1="cout_bob2", port_name_node2="cin_alice2")
    conn_cchanne3 = DirectConnection("CChannelConn_AB_3",
            ClassicalChannel("CChannel_A3->B3", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
            ClassicalChannel("CChannel_B3->A3", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchanne3, label="channel3",
                                port_name_node1="cout_bob", port_name_node2="cin_alice")
    # quantum_noise_modelに振幅減衰ノイズを指定
    # "quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False)
    # "quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False)
    # "quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False),
                                      "delay_model": FibreDelayModel(c=200e3)})
    network.add_connection(node_a, node_b, channel_to=qchannel, label="quantum",
                           port_name_node1="qout_bob", port_name_node2="qin_alice")
    
    # Link Alice ports:
    node_a.subcomponents["QSource_A1"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    node_a.subcomponents["QSource_A1"].ports["qout1"].connect(
        node_a.qmemory.ports["qin1"])
    node_a.subcomponents["QSource_A2"].ports["qout0"].connect(
        node_a.qmemory.ports["qin2"])
    node_a.subcomponents["QSource_A2"].ports["qout1"].connect(
        node_a.qmemory.ports["qin3"])
    node_a.qmemory.ports["qout"].forward_output(node_a.ports["qout_bob"])
    # Link Bob ports:
    node_b.add_ports([
        "qdispatch_in",
    ])
    node_b.ports["qin_alice"].forward_input(
        node_b.ports["qdispatch_in"]
    )
    return network

if __name__ == "__main__":
    network = network_setup()
    pb_example, dc = sim_setup(network.get_node("node_A"), network.get_node("node_B"), 3, np.pi/3, 0.2)
    pb_example.start()
    ns.sim_run()
    print("Fidelity of generated entanglement: {}".format(dc.dataframe["fidelity"].mean()))