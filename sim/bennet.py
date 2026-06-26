import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
import noise
from noise import AmplitudeNoiseModel, PhaseNoiseModel
import teleportation
from teleportation import InitStateProgram, BellMeasurement, Correction

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.qubitapi import fidelity, discard
from netsquid.qubits.ketstates import s00, b11
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
    def __init__(self, node, port, role, start_expression=None, msg_header="bennet", name=None):   # 初期化
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
        self._reprog = self._reverse_program()
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

    def _rotate_program(self):   # 回転操作（Aliceのみ）
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_Y, [q1])
        prog.apply(INSTR_Y, [q2])
        return prog
    
    def _measure_program(self):   # 測定（Alice & Bob）
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_CNOT, [q2, q1])
        prog.apply(INSTR_MEASURE, q1, output_key="M", inplace=False)
        return prog   # 測定結果をreturn

    def _reverse_program(self):   # 逆回転（Aliceのみ）
        prog = QuantumProgram(num_qubits=1)
        q1, = prog.get_qubit_indices(1)
        prog.apply(INSTR_Y, q1)
        return prog

    def run(self):
        #print(f"{self.name}:Start")
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            expr = yield cchannel_ready | qmemory_ready
            # 測定結果を受信した場合
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                #print(f"{self.name}: result {classical_message.items} received")
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
            # エンタングルメントを受信した場合
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                # エンタングルメントが保存されたメモリポジションを取得
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
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

    def _clear_qmem_positions(self):   # 失敗した場合、エンタングルメントを破棄
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]

    def _handle_new_qubit(self, memory_position):
        assert not self.node.qmemory.mem_positions[memory_position].is_empty
        
        if self._waiting_on_second_qubit:   # 2つ目のエンタングルメントが到着した場合
            assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
            assert memory_position != self._qmem_positions[0]
            self._qmem_positions[1] = memory_position
            self._waiting_on_second_qubit = False
            yield from self._node_do_bennet()   # 精製処理を行う
        else:   # 1つ目のエンタングルメントが到着した場合
            self.num_runs += 1
            #print(f"{self.name}: Sim {self.num_runs}")
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

    def _node_do_bennet(self):   # 精製処理
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        if self.role.upper() == "A": # Aliceの場合、回転操作を行う
            yield self.node.qmemory.execute_program(self._rotprog, [pos1, pos2])
        yield self.node.qmemory.execute_program(self._measprog, [pos1, pos2]) # 測定
        if self.role.upper() == "A": # Aliceの場合、回転操作を行う
            yield self.node.qmemory.execute_program(self._reprog, [pos2])
        self.local_meas_result = self._measprog.output["M"][0]
        self._qmem_positions[0] = None
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))   # 測定結果をBobに送信
        
    def _check_success(self):   # 測定結果の比較
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                self.send_signal(Signals.SUCCESS, [self._qmem_positions[1], self.num_runs])
                #print(f"{self.name}: SUCCESS / time: {sim_time()}")
                self.num_runs = 0
            else:
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
                #print(f"{self.name}: FAIL")
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

def network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=100, node_distance=200):
    network = Network("bennet_network")

    # ノード設定
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=6,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    state_sampler = StateSampler([ks.b11, ks.s00], probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
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
    # DepolarNoiseModelのtime_independentは、True->理論値を確認できる　False->時間依存なので現実に近くなる
    # "quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False)
    # "quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False)
    # "quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False),
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
        # エンタングルメント生成プロトコル
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=2, name="entangle_A"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", input_mem_pos=0,
                                           num_pairs=2, name="entangle_B"))
        # 精製処理プロトコル
        self.add_subprotocol(Bennet(node_a, node_a.ports["cout_bob"], role="A", name="bennet_A"))
        self.add_subprotocol(Bennet(node_b, node_b.ports["cin_alice"], role="B", name="bennet_B"))
        # テレポーテーションプロトコル
        self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
        self.add_subprotocol(Correction(node=node_b, name="teleport_B"))
        # エンタングルメント生成プロトコルの開始条件
        self.subprotocols["entangle_A"].start_expression = (
            self.subprotocols["entangle_A"].await_signal(self.subprotocols["bennet_A"], Signals.FAIL) |
                             self.subprotocols["entangle_A"].await_signal(self, Signals.WAITING))
        # 精製処理プロトコルの開始条件                        
        self.subprotocols["bennet_A"].start_expression = (
            self.subprotocols["bennet_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["bennet_B"].start_expression = (
            self.subprotocols["bennet_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(
                                                            self.subprotocols["bennet_A"], Signals.SUCCESS)
        self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(
                                                            self.subprotocols["bennet_B"], Signals.SUCCESS)
    
    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i}")
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            # 各ノードでのテレポーテーション処理が完了するまで待機
            yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS)) 
            signal_A = self.subprotocols["bennet_A"].get_signal_result(Signals.SUCCESS, self)
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

def sim_setup(node_a, node_b, num_runs):
    be_example = BennetExample(node_a, node_b, num_runs=num_runs)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result_en, result_tel = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        node_a.qmemory.discard(positions=[result_tel["pos_A0"]]) # 使用しているメモリを解放
        node_a.qmemory.discard(positions=[result_tel["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result_tel["pos_B"]])
        #print(qapi.reduced_dm([q_A, q_B]))
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)   # 忠実度を求める
        prob = 1 / result_en["runs"]
        return {"fidelity": f2, "pairs": result_en["pairs"], "probability": prob, "time": result_tel["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=be_example,
                                     event_type=Signals.SUCCESS.value))
    return be_example, dc

def run_experiment(node_distances):
    fidelity_data = pandas.DataFrame()
    for node_distance in node_distances:
        ns.sim_reset()
        network = network_setup(node_distance=node_distance)
        node_a = network.get_node("node_A")
        node_b = network.get_node("node_B")
        be_example, dc = sim_setup(node_a, node_b, 1000)
        be_example.start()
        ns.sim_run()
        df = dc.dataframe
        df['node_distance'] = node_distance
        fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data

def save_plot(datas, column, title, prefix):
    plot_style = {
        'kind': 'scatter',
        'grid': True,
        'title': title
    }
    data = datas.groupby("node_distance")[column].agg(
        **{column:'mean', 'sem':'sem'}).reset_index()
    save_dir = "./plots_test/bennet/node_distance/depolar"
    count1 = len([f for f in os.listdir(save_dir)
                 if f.startswith(prefix)])
    filename = f"{save_dir}/{prefix}_{count1 + 1}.png"
    data.plot(
        x='node_distance',
        y=column,
        yerr='sem',
        **plot_style
    )
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")
    count2 = len([f for f in os.listdir(save_dir)
                if f.startswith(column + " summary")])
    data[['node_distance', column]].to_csv(f"{save_dir}/{column} summary_{count2 + 1}.csv")

def create_plot_node():
    matplotlib.use('Agg')
    node_distances = [i for i in range(10, 1000, 50)]
    datas = run_experiment(node_distances)
    save_plot(
        datas,
        column="fidelity",
        title="Fidelity of the teleported quantum state with bennet\n(depolar_rate=100 Hz)",
        prefix="Bennet fidelity"
    )
    save_plot(
        datas,
        column="probability",
        title="Probability of success with bennet\n(depolar_rate=100 Hz)",
        prefix="Bennet probability"
    )
    save_plot(
        datas,
        column="pairs",
        title="Number of entanglement pairs used with bennet\n(depolar_rate=100 Hz)",
        prefix="Bennet pairs"
    )
    save_dir = "./plots_test/bennet/node_distance/depolar"
    count = len([f for f in os.listdir(save_dir)
                 if f.startswith("Bennet result")])
    datas.to_csv(f"{save_dir}/Bennet result_{count + 1}.csv")

def run_experiment_noise(noise_rate):
    fidelity_data = pandas.DataFrame()
    for noise in noise_rate:
        ns.sim_reset()
        network = network_setup(depolar_rate=noise)
        node_a = network.get_node("node_A")
        node_b = network.get_node("node_B")
        be_example, dc = sim_setup(node_a, node_b, 1000)
        be_example.start()
        ns.sim_run()
        df = dc.dataframe
        df['depolar_rate'] = noise
        fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data

def save_plot_noise(datas, column, title, prefix):
    plot_style = {
        'kind': 'scatter',
        'grid': True,
        'title': title
    }
    data = datas.groupby("depolar_rate")[column].agg(
        **{column:'mean', 'sem':'sem'}).reset_index()
    save_dir = "./plots_test/bennet/noise/depolar"
    count1 = len([f for f in os.listdir(save_dir)
                 if f.startswith(prefix)])
    filename = f"{save_dir}/{prefix}_{count1 + 1}.png"
    data.plot(
        x='depolar_rate',
        y=column,
        yerr='sem',
        **plot_style
    )
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")
    count2 = len([f for f in os.listdir(save_dir)
                if f.startswith(column + " summary")])
    data[['depolar_rate', column]].to_csv(f"{save_dir}/{column} summary_{count2 + 1}.csv")

def create_plot_noise():
    matplotlib.use('Agg')
    noise_rate = [i for i in range(0, 1500, 100)]
    datas = run_experiment_noise(noise_rate)
    save_plot_noise(
        datas,
        column="fidelity",
        title="Fidelity of the teleported quantum state with bennet\n(node_distance=200 km)",
        prefix="Bennet fidelity"
    )
    save_plot_noise(
        datas,
        column="probability",
        title="Probability of success with bennet\n(node_distance=200 km)",
        prefix="Bennet probability"
    )
    save_plot_noise(
        datas,
        column="pairs",
        title="Number of entanglement pairs used with bennet\n(node_distance=200 km)",
        prefix="Bennet pairs"
    )
    save_dir = "./plots_test/bennet/noise/depolar"
    count = len([f for f in os.listdir(save_dir)
                 if f.startswith("Bennet result")])
    datas.to_csv(f"{save_dir}/Bennet result_{count + 1}.csv")

if __name__ == "__main__":
    #network = network_setup()
    #be_example, dc = sim_setup(network.get_node("node_A"), network.get_node("node_B"), num_runs=1)
    #be_example.start()
    #ns.sim_run()
    #print("Average fidelity of generated entanglement with bennet: {}".format(dc.dataframe["fidelity"].mean()))
    #print("Average resource with bennet: {}".format(dc.dataframe["pairs"].mean()))
    #print("Average probability of success with bennet: {}".format(dc.dataframe["probability"].mean()))
    create_plot_node()