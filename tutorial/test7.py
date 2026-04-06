import netsquid as ns
import pandas
import pydynaa
from netsquid.components import ClassicalChannel, Channel, QuantumChannel
from netsquid.components.models import FibreDelayModel
from netsquid.components.models.delaymodels import FixedDelayModel
from netsquid.components.models.qerrormodels import DephaseNoiseModel, DepolarNoiseModel
import netsquid.components.instructions as instr
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.nodes import Node
from netsquid.nodes.connections import Connection
from netsquid.nodes.network import Network
from netsquid.components.qprogram import QuantumProgram
from netsquid.protocols import NodeProtocol, Signals
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits import StateSampler
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.ketstates as ks
from netsquid.examples.teleportation import example_network_setup
from netsquid.util.datacollector import DataCollector

class EntanglingConnection(Connection): #エンタングル状態の量子ビットの生成＆ポートの接続
    def __init__(self, length, source_frequency, name="EntanglingConnection"):
        super().__init__(name=name)
        qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2,
                          timing_model=FixedDelayModel(delay=1e9 / source_frequency),
                          status=SourceStatus.INTERNAL)
        self.add_subcomponent(qsource, name="qsource")
        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])

class ClassicalConnection(Connection): #古典チャネルの設定＆接続
    def __init__(self, length, name="ClassicalConnection"):
        super().__init__(name=name)
        self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length,
                                               models={"delay_model": FibreDelayModel()}),
                              forward_input=[("A", "send")],
                              forward_output=[("B", "recv")])

def create_processor(depolar_rate, dephase_rate): #Alice&Bobのプロセッサを作成
    # We'll give both Alice and Bob the same kind of processor
    measure_noise_model = DephaseNoiseModel(dephase_rate=dephase_rate,
                                            time_independent=True)
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=3, parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0, 1]),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True, topology=[(0, 1)]),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[0],
                            quantum_noise_model=measure_noise_model, apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[1])
    ]
    memory_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    processor = QuantumProcessor("quantum_processor", num_positions=2,
                                 memory_noise_models=[memory_noise_model] * 2,
                                 phys_instructions=physical_instructions)
    return processor

def example_network_setup(node_distance=4e-3, depolar_rate=1e7, dephase_rate=0.2): #ネットワークの構築
    # Setup nodes Alice and Bob with quantum processor:
    alice = Node("Alice", qmemory=create_processor(depolar_rate, dephase_rate))
    bob = Node("Bob", qmemory=create_processor(depolar_rate, dephase_rate))
    # Create a network
    network = Network("Teleportation_network")
    network.add_nodes([alice, bob])
    # Setup classical connection between nodes:
    c_conn = ClassicalConnection(length=node_distance)
    network.add_connection(alice, bob, connection=c_conn, label="classical",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    # Setup entangling connection between nodes:
    source_frequency = 4e4 / node_distance
    q_conn = EntanglingConnection(
        length=node_distance, source_frequency=source_frequency)
    port_ac, port_bc = network.add_connection(
        alice, bob, connection=q_conn, label="quantum",
        port_name_node1="qin_charlie", port_name_node2="qin_charlie")
    alice.ports[port_ac].forward_input(alice.qmemory.ports['qin1'])
    bob.ports[port_bc].forward_input(bob.qmemory.ports['qin0'])
    return network

#Setting up the simulation
class InitStateProgram(QuantumProgram): #伝送する量子ビットの作成
    default_num_qubits = 1

    def program(self):
        q1, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_S, q1)
        yield self.run()

class BellMeasurementProgram(QuantumProgram): #Aliceが行う測定
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
        yield self.run()

class BellMeasurementProtocol(NodeProtocol): #測定の実行＆送信
    def run(self):
        qubit_initialised = False
        entanglement_ready = False
        qubit_init_program = InitStateProgram()
        measure_program = BellMeasurementProgram()
        self.node.qmemory.execute_program(qubit_init_program)
        while True:
            expr = yield (self.await_program(self.node.qmemory) |
                          self.await_port_input(self.node.ports["qin_charlie"]))
            if expr.first_term.value:
                qubit_initialised = True
            else:
                entanglement_ready = True
            if qubit_initialised and entanglement_ready:
                # Once both qubits arrived, do BSM program and send to Bob
                yield self.node.qmemory.execute_program(measure_program)
                m1, = measure_program.output["M1"]
                m2, = measure_program.output["M2"]
                self.node.ports["cout_bob"].tx_output((m1, m2))
                self.send_signal(Signals.SUCCESS)
                qubit_initialised = False
                entanglement_ready = False
                self.node.qmemory.execute_program(qubit_init_program)

class CorrectionProtocol(NodeProtocol): #受信＆適当な操作を加える
    def run(self):
        port_alice = self.node.ports["cin_alice"]
        port_charlie = self.node.ports["qin_charlie"]
        entanglement_ready = False
        meas_results = None
        while True:
            # Wait for measurement results of Alice or qubit from Charlie to arrive
            expr = yield (self.await_port_input(port_alice) |
                          self.await_port_input(port_charlie))
            if expr.first_term.value:  # If measurements from Alice arrived
                meas_results, = port_alice.rx_input().items
            else:
                entanglement_ready = True
            if meas_results is not None and entanglement_ready:
                # Do corrections (blocking)
                if meas_results[0] == 1:
                    self.node.qmemory.execute_instruction(instr.INSTR_Z)
                    yield self.await_program(self.node.qmemory)
                if meas_results[1] == 1:
                    self.node.qmemory.execute_instruction(instr.INSTR_X)
                    yield self.await_program(self.node.qmemory)
                self.send_signal(Signals.SUCCESS, 0)
                entanglement_ready = False
                meas_results = None

def example_sim_setup(node_A, node_B): #シミュレーションの設定
    def collect_fidelity_data(evexpr):
        protocol = evexpr.triggered_events[-1].source
        mem_pos = protocol.get_signal_result(Signals.SUCCESS)
        qubit, = protocol.node.qmemory.pop(mem_pos)
        fidelity = qapi.fidelity(qubit, ns.y0, squared=True)
        qapi.discard(qubit)
        return {"fidelity": fidelity}

    protocol_alice = BellMeasurementProtocol(node_A)
    protocol_bob = CorrectionProtocol(node_B)
    dc = DataCollector(collect_fidelity_data)
    dc.collect_on(pydynaa.EventExpression(source=protocol_bob,
                                          event_type=Signals.SUCCESS.value))
    return protocol_alice, protocol_bob, dc
    
def run_experiment(num_runs, depolar_rates, distance=4e-3, dephase_rate=0.0): #シミュレーションを実行
    fidelity_data = pandas.DataFrame()
    for depolar_rate in depolar_rates:
        ns.sim_reset()
        network = example_network_setup(distance, depolar_rate, dephase_rate)
        node_a = network.get_node("Alice")
        node_b = network.get_node("Bob")
        protocol_alice, protocol_bob, dc = example_sim_setup(node_a, node_b)
        protocol_alice.start()
        protocol_bob.start()
        q_conn = network.get_connection(node_a, node_b, label="quantum")
        cycle_runtime = (q_conn.subcomponents["qsource"].subcomponents["internal_clock"]
                         .models["timing_model"].delay)
        ns.sim_run(cycle_runtime * num_runs + 1)
        df = dc.dataframe
        df['depolar_rate'] = depolar_rate
        fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data                                    

def create_plot(): #グラフをプロット
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('Agg')
    depolar_rates = [1e6 * i for i in range(0, 200, 10)]
    fidelities = run_experiment(num_runs=1000, distance=4e-3,
                                depolar_rates=depolar_rates, dephase_rate=0.0)
    plot_style = {'kind': 'scatter', 'grid': True,
                  'title': "Fidelity of the teleported quantum state"}
    data = fidelities.groupby("depolar_rate")['fidelity'].agg(
        fidelity='mean', sem='sem').reset_index()
    save_dir = "./plots"
    existing_files = len([f for f in os.listdir(save_dir) if f.startswith("Fidelity")])
    filename = f"{save_dir}/Fidelity_{existing_files + 1}.png"
    data.plot(x='depolar_rate', y='fidelity', yerr='sem', **plot_style)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

import os
create_plot()