import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
import math
from matplotlib import pyplot as plt
import noise
from noise import AmplitudeNoiseModel, PhaseNoiseModel
import teleportation
from teleportation import InitStateProgram, BellMeasurement, Correction

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

class Protect(NodeProtocol):   # Alice側のプロトコル
    def __init__(self, node, port, start_expression=None, msg_header="protect", omega=np.pi/3, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "ProtectNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.num_runs = 0
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
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
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    #print(f"{self.name}: Bob's result received {classical_message}")
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                        event=expr.second_term.triggered_events[0], receiver=self) # エンタングルメントが保存されたメモリポジションを取得
                #print(f"{self.name}: Entanglement received at {ready_signal.result} / time: {sim_time()}")      
                self._qmem_positions[0] = ready_signal.result["mem_pos0"]
                self._qmem_positions[1] = ready_signal.result["mem_pos1"]
                yield from self._handle_qubit_rx()

    def start(self):
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        return super().start()

    def _handle_qubit_rx(self):
        self.num_runs += 1
        #print(f"{self.name}: Sim {self.num_runs}")
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [pos2], 
                                                       meas_operators=self.wmeas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        self.local_meas_result = output["instr"][0]
        #print(f"{self.name}: Result = {self.local_meas_result}")
        self.local_qcount += 1
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result], header=self.header))
        if self.local_meas_result == 1:
            #print(f"{self.name}: Flip operation")
            if self.node.qmemory.busy:
                yield self.await_program(self.node.qmemory)
            self.node.qmemory.execute_instruction(INSTR_X, [pos2])
        qubit = self.node.qmemory.pop(positions=pos2)
        #print(f"{self.name}: {self.node.qmemory.used_positions}")
        self._qmem_positions[1] = None
        #self._check_success()

    def _handle_cchannel_rx(self):
        if (self._qmem_positions is not None and
                self.node.qmemory.mem_positions[self._qmem_positions[0]].in_use):
            self._check_success()

    def _check_success(self):
        #print(f"{self.name}: Remote result is {self.remote_meas_result}")
        if self.remote_meas_result == 1:
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)
            #print(f"{self.name}: FAIL")
            self.local_meas_result = None
            self.remote_meas_result = None
        else:
            self.send_signal(Signals.SUCCESS, [self._qmem_positions[0], self.num_runs])
            #print(f"{self.name}: SUCCESS")
            self.num_runs = 0

    def _handle_fail(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]
        

class RWMeasure(NodeProtocol):   # Bob側のプロトコル
    def __init__(self, node, port_c, port_q, start_expression=None, msg_header="protect", theta=0.2, name=None):
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
        while True:
            cchannel_ready = self.await_port_input(self.port_c)
            qmemory_ready = self.await_port_input(self.port_q)
            expr = yield cchannel_ready | qmemory_ready
            if expr.first_term.value:
                classical_message = self.port_c.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    #print(f"{self.name}: Alice's result received {classical_message}")
            elif expr.second_term.value:
                #print(f"{self.name}: {self.node.qmemory.used_positions}")
                self._qmem_pos = self.node.qmemory.used_positions
                #print(f"{self.name}: Entanglement arrived at {self._qmem_pos}")
                #print(f"{self.name}: Remote result = {self.remote_meas_result}")
                if self.remote_meas_result is not None:
                    yield from self._handle_qubit_rx()
    
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_result = None
        self.remote_meas_result = None
        return super().start()

    def stop(self):
        super().stop()

    def _handle_qubit_rx(self):
        self.local_qcount += 1
        #print(f"{self.name}: Local qcount is {self.local_qcount}")
        pos = self._qmem_pos[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        if self.remote_meas_result == 1:
            #print(f"{self.name}: Remote result = 1 -> Flip operation")
            self.node.qmemory.execute_instruction(INSTR_X, [pos])
            if self.node.qmemory.busy:
                yield self.await_program(self.node.qmemory)
            output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [pos], meas_operators=self.rwmeas_ops1)
            self.local_meas_result = output[0]["instr"][0]
            #print(f"{self.name}: Result = {output}")
        else:
            #print(f"{self.name}: Remote result = 0")
            output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [pos], meas_operators=self.rwmeas_ops0)[0]
            self.local_meas_result = output["instr"][0]
            #print(f"{self.name}: Result = {self.local_meas_result}")
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        self.port_c.tx_output(Message([self.local_qcount, self.local_meas_result], header=self.header))
        self._check_success()

    def _check_success(self):
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_result == 0):
            #print(f"{self.name}: SUCCESS")
            self.send_signal(Signals.SUCCESS, self._qmem_pos[0])
            self.remote_meas_result = None
        elif self.local_meas_result == 0 and self.local_qcount > self.remote_qcount:
            pass
        else:
            self._handle_fail()
            #print(f"{self.name}: FAIL")
            self.send_signal(Signals.FAIL, self.local_qcount)
            self.local_meas_result = None
            self.remote_meas_result = None
    
    def _handle_fail(self):
        positions = [pos for pos in self._qmem_pos if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_pos = [None] * len(self._qmem_pos)

        

def network_setup(source_delay=1e5, source_fidelity_sq=0.8, damp_rate=1400, node_distance=200):
    network = Network("wmeasure_network")

    # ノード設定
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=6,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    state_sampler = StateSampler([ks.b00, ks.s00], probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    source_frequency = 4e4 / node_distance
    node_a.add_subcomponent(QSource("QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor("QuantumMemory_B", num_positions=6,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能

    # チャネル設定
    conn_cchannel = DirectConnection("CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel,
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    # quantum_noise_modelに振幅減衰ノイズを指定
    # "quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False)
    # "quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False)
    # "quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False)
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

class ProtectExample(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs, omega, theta):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Protect example")
        self.num_runs = num_runs
        # エンタングルメント生成プロトコル
        self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A", input_mem_pos0=0,
                                           input_mem_pos1=1, num_pairs=1, name="entangle_A"))
        # 保護処理プロトコル
        self.add_subprotocol(Protect(node_a, node_a.ports["cout_bob"], omega=np.pi/3, name="protect_A"))
        self.add_subprotocol(RWMeasure(node_b, node_b.ports["cin_alice"],
                             node_b.ports["qin_alice"], theta=0.2, name="rwmeasure_B"))
        # テレポーテーションプロトコル
        self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
        self.add_subprotocol(Correction(node=node_b, name="teleport_B"))
        # エンタングルメント生成プロトコルの開始条件
        self.subprotocols["entangle_A"].start_expression = (
                             self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING) |
                             self.subprotocols["entangle_A"].await_signal(self.subprotocols["protect_A"], Signals.FAIL))
        # 保護処理プロトコルの開始条件                        
        self.subprotocols["protect_A"].start_expression = (
            self.subprotocols["protect_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["rwmeasure_B"].start_expression = (
            self.subprotocols["rwmeasure_B"].await_signal(self, Signals.WAITING) |
            self.subprotocols["rwmeasure_B"].await_signal(self.subprotocols["protect_A"], Signals.FAIL))
        # テレポーテーションプロトコルの開始条件
        self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(
                                                            self.subprotocols["protect_A"], Signals.SUCCESS)
        self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(
                                                            self.subprotocols["rwmeasure_B"], Signals.SUCCESS)

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i}")
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS) &
                    self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS))
            signal_A = self.subprotocols["protect_A"].get_signal_result(Signals.SUCCESS, self)
            result_en = {
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
                "runs": signal_A[1]
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
    pro_example = ProtectExample(node_a, node_b, num_runs, omega, theta)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result_en, result_tel = protocol.get_signal_result(Signals.SUCCESS)
        #print(result_tel)
        # Record fidelity
        node_a.qmemory.discard(positions=[result_tel["pos_A0"]]) # popにより、使用しているメモリを解放
        node_a.qmemory.discard(positions=[result_tel["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result_tel["pos_B"]])
        #print(qapi.reduced_dm([q_A, q_B]))
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)
        prob = 1 / result_en["runs"]
        return {"fidelity": f2, "pairs": result_en["pairs"], "probability": prob, "time": result_tel["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=pro_example,
                                     event_type=Signals.SUCCESS.value))
    return pro_example, dc

def run_experiment(var_o, var_t):
    fidelity_data = pandas.DataFrame()
    for omega in var_o:
        for theta in var_t:
            ns.sim_reset()
            network = network_setup()
            node_a = network.get_node("node_A")
            node_b = network.get_node("node_B")
            pro_example, dc = sim_setup(node_a, node_b, 1000, omega, theta)
            pro_example.start()
            ns.sim_run()
            df = dc.dataframe
            df['omega'] = omega
            df['theta'] = theta
            fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data

def save_heatmap(dataframe, value_col, title, colorbar_label, filename_prefix, var_o, var_t):
    data = dataframe.groupby(["omega", "theta"])[value_col].mean().reset_index()
    heatmap_data = data.pivot(index='theta', columns='omega', values=value_col)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, origin='lower', aspect='auto',
                    extent=[
                        min(var_o), max(var_o),
                        min(var_t), max(var_t)
                    ]
    )
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel(r'WM strength $\omega$')
    plt.ylabel(r'WMR strength $\theta$')
    plt.title(title)
    save_dir = "./plots_test/protect/noise/1400/amplitude"
    count1 = len([f for f in os.listdir(save_dir) if f.startswith(filename_prefix)])
    filename = f"{save_dir}/{filename_prefix}_{count1 + 1}.png"
    plt.savefig(filename)
    plt.close()
    if value_col == "fidelity":
        max_fid = data["fidelity"].max()
        best_rows = data[data["fidelity"] == max_fid]
        print("Optimal value is")
        print(best_rows)
    print(f"Plot saved as {filename}")
    count2 = len([f for f in os.listdir(save_dir)
                if f.startswith(value_col + " summary")])
    data[['omega', 'theta', value_col]].to_csv(f"{save_dir}/{value_col} summary_{count2 + 1}.csv")

def create_plot():
    matplotlib.use('Agg')
    var_o = [i for i in np.arange(0.0, np.pi/2, np.pi/14)]
    var_t = [i for i in np.arange(0.0, 1.0, 0.1)]
    datas = run_experiment(var_o, var_t)
    # 忠実度
    save_heatmap(
        datas,
        value_col="fidelity",
        title="Fidelity Heatmap with protection\n(damp_rate=1400 Hz, node_distance=200 km)",
        colorbar_label="Average Fidelity",
        filename_prefix="Protect fidelity",
        var_o=var_o,
        var_t=var_t
    )
    # 成功確率
    save_heatmap(
        datas,
        value_col="probability",
        title="Success Probability Heatmap with protection\n(damp_rate=1400 Hz, node_distance=200 km)",
        colorbar_label="Average Success Probability",
        filename_prefix="Protect probability",
        var_o=var_o,
        var_t=var_t
    )
    # エンタングルメントペア消費数
    save_heatmap(
        datas,
        value_col="pairs",
        title="Entanglement Pair Usage Heatmap with protection\n(damp_rate=1400 Hz, node_distance=200 km)",
        colorbar_label="Average Number of Pairs",
        filename_prefix="Protect pairs",
        var_o=var_o,
        var_t=var_t
    )
    save_dir = "./plots_test/protect/noise/1400/amplitude"
    count = len([f for f in os.listdir(save_dir) if f.startswith("Protect result")])
    datas.to_csv(f"{save_dir}/Protect result_{count + 1}.csv")
        
if __name__ == "__main__":
    #network = network_setup()
    #pro_example, dc = sim_setup(network.get_node("node_A"), network.get_node("node_B"), 1, np.pi/3, 0.2)
    #pro_example.start()
    #ns.sim_run()
    #print("Average fidelity of generated entanglement with protection: {}".format(dc.dataframe["fidelity"].mean()))
    #print("Average resource with protection: {}".format(dc.dataframe["pairs"].mean()))
    #print("Average probability of success with protection: {}".format(dc.dataframe["probability"].mean()))
    create_plot()