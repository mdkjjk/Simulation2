# シミュレーション：精製処理を適用しない場合の伝送忠実度の測定

import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
import noise
from noise import AmplitudeNoiseModel, PhaseNoiseModel

# 必要となるパッケージのインポート
import netsquid.components.instructions as instr
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.examples.entanglenodes import EntangleNodes
from pydynaa import EventExpression


class InitStateProgram(QuantumProgram): # 伝送する量子ビットを生成するプログラム
    default_num_qubits = 1

    def program(self):
        q1, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_S, q1)
        yield self.run()


class BellMeasurement(NodeProtocol): # 量子テレポーテーションにおけるAlice側での処理を行うプロトコル
    def __init__(self, node, port, name=None):
        super().__init__(node, name)
        self.port = port
        self._qmem_pos0 = None
        self._qmem_pos1 = None

    def start(self):
        super().start()
        if self.start_expression is not None and not isinstance(self.start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(self.start_expression)))

    def run(self):
        qubit_initialised = False
        entanglement_ready = False
        qubit_init_program = InitStateProgram()
        while True:
            expr_port = self.start_expression
            yield expr_port # エンタングルメントの到着を待機
            entanglement_ready = True
            source_protocol = expr_port.atomic_source
            ready_signal = source_protocol.get_signal_by_event(event=expr_port.triggered_events[0], receiver=self)
            #print(ready_signal.result)
            self._qmem_pos1 = ready_signal.result[0] # エンタングルメントが保存された量子メモリポジションを取得
            print(f"{self.name}: Entanglement received at {self._qmem_pos1} / time: {sim_time()}")

            # 空いている量子メモリポジションで伝送する量子ビットを生成
            self._qmem_pos0 = self.node.qmemory.unused_positions[0]
            self.node.qmemory.execute_program(qubit_init_program, qubit_mapping=[self._qmem_pos0])
            expr_signal = self.await_program(self.node.qmemory)
            yield expr_signal # 伝送する量子ビットの生成が完了するまで待機
            qubit_initialised = True
            print(f"{self.name}: Initqubit received at {self._qmem_pos0} / time: {sim_time()}")

            # 上記の2つが完了した場合
            if qubit_initialised and entanglement_ready: 
                self.node.qmemory.operate(ns.CNOT, [self._qmem_pos0, self._qmem_pos1]) # CNOTゲート適用
                self.node.qmemory.operate(ns.H, self._qmem_pos0) # Hゲート適用
                m, _ = self.node.qmemory.measure([self._qmem_pos0, self._qmem_pos1]) # 測定
                self.port.tx_output(m) # 測定結果をBobへ送信
                result = {"pos_A0": self._qmem_pos0,
                          "pos_A1": self._qmem_pos1,}
                self.send_signal(Signals.SUCCESS, result) # プロトコルの成功シグナルと使用しているメモリポジションを送信
                #print(f"{self.name}: Finish / time: {sim_time()}")
                qubit_initialised = False
                entanglement_ready = False


class Correction(NodeProtocol): # 量子テレポーテーションにおけるBob側での処理を行うプロトコル
    def __init__(self, node, start_expression=None, name=None):
        super().__init__(node, name)
        self.start_expression = start_expression
        self._qmem_pos = None

    def run(self):
        port_alice = self.node.ports["cin_alice"]
        entanglement_ready = False
        meas_results = None
        while True:
            expr_signal = self.start_expression
            expr = yield (self.await_port_input(port_alice) | expr_signal) # エンタングルメントの到着 or Aliceの測定結果の到着を待機

            # Aliceの測定結果が到着した場合
            if expr.first_term.value:
                meas_results = port_alice.rx_input().items # 測定結果を取得
                print(f"{self.name}: Result: {meas_results} / time: {sim_time()}")
            # エンタングルメントが到着した場合
            else:
                entanglement_ready = True
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(event=expr.second_term.triggered_events[-1], receiver=self)
                self._qmem_pos = ready_signal.result[0] # エンタングルメントが保存された量子メモリポジションを取得
                print(f"{self.name}: Entanglement received at {self._qmem_pos} / time: {sim_time()}")

            # 上記の2つが完了した場合
            if meas_results is not None and entanglement_ready:
                # 測定結果に応じて、Z,Xゲートを適用
                if meas_results[0] == 1:
                    self.node.qmemory.execute_instruction(instr.INSTR_Z, [self._qmem_pos])
                if meas_results[1] == 1:
                    self.node.qmemory.execute_instruction(instr.INSTR_X, [self._qmem_pos])
                self.send_signal(Signals.SUCCESS, self._qmem_pos) # プロトコルの成功シグナルと使用しているメモリポジションを送信
                #print(f"{self.name}: Teleport success / time: {sim_time()}")
                entanglement_ready = False
                meas_results = None


class Example(LocalProtocol): # シミュレーション全体のプロトコル
    def __init__(self, node_a, node_b, num_runs):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="example")
        self.num_runs = num_runs
        # 各ノードで実行するサブプロトコルを追加
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=1, name="entangle_A"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", input_mem_pos=0,
                                           num_pairs=1, name="entangle_B"))
        self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
        self.add_subprotocol(Correction(node=node_b, name="teleport_B"))

        # 各サブプロトコルの開始条件を設定
        self.subprotocols["entangle_A"].start_expression = self.subprotocols["entangle_A"].await_signal(
                                                            self, Signals.WAITING)
        self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(
                                                            self.subprotocols["entangle_A"], Signals.SUCCESS)
        self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(
                                                            self.subprotocols["entangle_B"], Signals.SUCCESS)

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs): # self.num_runsの回数シミュレーションを試行
            #print(f"Simulation {i} Start")
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING) # WAITINGシグナルにより、エンタングルメント生成が開始

            yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS)) # 各ノードでのテレポーテーション処理が完了するまで待機
            result_A = self.subprotocols["teleport_A"].get_signal_result(Signals.SUCCESS, self)
            signal_B = self.subprotocols["teleport_B"].get_signal_result(Signals.SUCCESS, self)
            result = {
                "pos_A0": result_A["pos_A0"],
                "pos_A1": result_A["pos_A1"],
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs
            }
            self.send_signal(Signals.SUCCESS, result) # プロトコルの成功シグナルと使用しているメモリポジションを送信
            #print(f"Simulation {i} Finish")


def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=1500,
                          node_distance=30): # 量子ネットワークの構築
    network = Network("network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"]) # ノードの追加
    # 各ノードにコンポーネントを追加
    # ノードA: QuantumProcessor(量子メモリ), QSource(量子ソース)
    # ノードB: QuantumProcessor(量子メモリ)
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0))) # メモリ数や、メモリに働くノイズを設定可能
    state_sampler = StateSampler([ks.b00, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq]) # 量子ソースで生成される量子状態の設定
    node_a.add_subcomponent(QSource(
        "QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0)))
    node_a.add_ports(["cout_bob_dis", "cout_bob_fil"])
    node_b.add_ports(["cin_alice_dis", "cin_alice_fil"])

    # 古典チャネルを接続
    cchannel = DirectConnection("CChannelConn_tel", ClassicalChannel("CChannel_dis_A->B", length=node_distance,
                                models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=cchannel, label="tereport",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")

    # 量子チャネルを接続
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate),
                                      "delay_model": FibreDelayModel(c=200e3)},
                              depolar_rate=0)
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charlie")

    # 量子ソースのポートを各ノードのメモリと接続
    node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    return network


def example_sim_setup(node_a, node_b, num_runs): # シミュレーションの設定
    example = Example(node_a, node_b, num_runs=num_runs)

    def record_run(evexpr): # 1回のシミュレーションが成功したら実行
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS) # Exampleプロトコルの結果を取得
        node_a.qmemory.pop(positions=[result["pos_A0"]]) # popにより、使用しているメモリを解放
        node_a.qmemory.pop(positions=[result["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity(q_B, ks.y0, squared=True) # 忠実度の計算
        return {"F2": f2, "pairs": result["pairs"], "time": result["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=example,
                                     event_type=Signals.SUCCESS.value))
    return example, dc


def run_experiment(node_distances): # シミュレーションを実行するための関数
    fidelity_data = pandas.DataFrame()
    for node_distance in node_distances:
        ns.sim_reset()
        network = example_network_setup(node_distance=node_distance) # ネットワーク構築
        node_a = network.get_node("node_A")
        node_b = network.get_node("node_B")
        example, dc = example_sim_setup(node_a, node_b, 100) # シミュレーション設定
        example.start()
        ns.sim_run() # シミュレーション実行
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
        mean='mean', sem='sem').reset_index()
    save_dir = "./plots_test"
    count = len([f for f in os.listdir(save_dir)
                 if f.startswith(prefix)])
    filename = f"{save_dir}/{prefix}_{count + 1}.png"
    data.plot(
        x='node_distance',
        y='mean',
        yerr='sem',
        **plot_style
    )
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")

def create_plot():
    matplotlib.use('Agg')
    node_distances = [1 + i for i in range(0, 100, 5)]
    datas = run_experiment(node_distances)
    save_plot(
        datas,
        column="F2",
        title="Fidelity of the teleported quantum state with Teleport",
        prefix="Teleport fidelity"
    )
    save_dir = "./plots_test"
    count = len([f for f in os.listdir(save_dir) if f.startswith("Teleport result")])
    datas.to_csv(f"{save_dir}/Teleport result_{count + 1}.csv")


if __name__ == "__main__":
    #network = example_network_setup()
    #example, dc = example_sim_setup(network.get_node("node_A"),network.get_node("node_B"),num_runs=1)
    #example.start()
    #ns.sim_run()
    #print("Average fidelity of received qubit: {}".format(dc.dataframe["F2"].mean()))
    create_plot()
