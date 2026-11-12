import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
from noise import AmplitudeNoiseModel, PhaseNoiseModel
from teleportation import BellMeasurement, Correction
from filter import Filter
from wmb import Bennet
from entanglenodes import EntangleNodes

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection

ns.set_qstate_formalism(QFormalism.DM)

class FilteringBennet(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs, epsilon=0.9):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Filtering&Bennet")
        self._epsilon = epsilon
        self.num_runs = num_runs
        # エンタングルメント生成プロトコル
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", qsource_name="QSource_A1", input_mem_pos=0,
                                           num_pairs=1, name="entangle_A1"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", qsource_name="QSource_A1", input_mem_pos=0,
                                           num_pairs=1, name="entangle_B1"))
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", qsource_name="QSource_A2", input_mem_pos=1,
                                           num_pairs=1, name="entangle_A2"))
        self.add_subprotocol(EntangleNodes(node=node_b, role="receiver", qsource_name="QSource_A2", input_mem_pos=1,
                                           num_pairs=1, name="entangle_B2"))
        # フィルタープロトコル
        self.add_subprotocol(Filter(node_a, node_a.ports["cout_bob1"], epsilon=epsilon, name="filter_A1"))
        self.add_subprotocol(Filter(node_a, node_a.ports["cout_bob2"], epsilon=epsilon, name="filter_A2"))                    
        self.add_subprotocol(Filter(node_b, node_b.ports["cin_alice1"], epsilon=epsilon, name="filter_B1"))
        self.add_subprotocol(Filter(node_b, node_b.ports["cin_alice2"], epsilon=epsilon, name="filter_B2"))
        # 精製処理プロトコル
        self.add_subprotocol(Bennet(node_a, node_a.ports["cout_bob"], role="A", name="bennet_A'"))
        self.add_subprotocol(Bennet(node_b, node_b.ports["cin_alice"], role="B", name="bennet_B'"))    
        # テレポーテーションプロトコル
        self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
        self.add_subprotocol(Correction(node=node_b, name="teleport_B"))    
                                    
        # エンタングルメント生成プロトコルの開始条件
        self.subprotocols["entangle_A1"].start_expression = (
                    self.subprotocols["entangle_A1"].await_signal(self, Signals.WAITING) |
                    self.subprotocols["entangle_A1"].await_signal(self.subprotocols["filter_A1"], Signals.FAIL) |
                    self.subprotocols["entangle_A1"].await_signal(self.subprotocols["bennet_A'"], Signals.FAIL))
        self.subprotocols["entangle_A2"].start_expression = (
                    self.subprotocols["entangle_A2"].await_signal(self, Signals.WAITING) |
                    self.subprotocols["entangle_A2"].await_signal(self.subprotocols["filter_A2"], Signals.FAIL) |
                    self.subprotocols["entangle_A2"].await_signal(self.subprotocols["bennet_A'"], Signals.FAIL))
                                                             
        # フィルタープロトコルの開始条件
        self.subprotocols["filter_A1"].start_expression = (
                    self.subprotocols["filter_A1"].await_signal(self.subprotocols["entangle_A1"], Signals.SUCCESS))
        self.subprotocols["filter_B1"].start_expression = (
                    self.subprotocols["filter_B1"].await_signal(self.subprotocols["entangle_B1"], Signals.SUCCESS))
        self.subprotocols["filter_A2"].start_expression = (
                    self.subprotocols["filter_A2"].await_signal(self.subprotocols["entangle_A2"], Signals.SUCCESS))
        self.subprotocols["filter_B2"].start_expression = (
                    self.subprotocols["filter_B2"].await_signal(self.subprotocols["entangle_B2"], Signals.SUCCESS))   
        
        # 精製処理プロトコルの開始条件                        
        self.subprotocols["bennet_A'"].start_expression = (
            self.subprotocols["bennet_A'"].await_signal(self.subprotocols["filter_A1"], Signals.SUCCESS) &
            self.subprotocols["bennet_A'"].await_signal(self.subprotocols["filter_A2"], Signals.SUCCESS))
                                                        
        self.subprotocols["bennet_B'"].start_expression = (
            self.subprotocols["bennet_B'"].await_signal(self.subprotocols["filter_A1"], Signals.SUCCESS) &
            self.subprotocols["bennet_B'"].await_signal(self.subprotocols["filter_A2"], Signals.SUCCESS))       

        # テレポーテーションプロトコルの開始条件
        self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(
                                                            self.subprotocols["bennet_A'"], Signals.SUCCESS)
        self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(
                                                            self.subprotocols["bennet_B'"], Signals.SUCCESS)  
                
        
    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i}: Start")
            start_time = sim_time()
            self.subprotocols["entangle_A1"].entangled_pairs = 0
            self.subprotocols["entangle_A2"].entangled_pairs = 0
            self.subprotocols["filter_A1"].num_runs = 0
            self.subprotocols["filter_A2"].num_runs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS))
            filter1 = self.subprotocols["filter_A1"].get_signal_result(Signals.SUCCESS,self)
            filter2 = self.subprotocols["filter_A2"].get_signal_result(Signals.SUCCESS,self)
            bennet = self.subprotocols["bennet_A'"].get_signal_result(Signals.SUCCESS,self)
            result_en = {
                "runs_f1": filter1[1],
                "runs_f2": filter2[1],
                "runs_b": bennet[1],
                "pairs1": self.subprotocols["entangle_A1"].entangled_pairs,
                "pairs2": self.subprotocols["entangle_A2"].entangled_pairs,
            }
            #print(result_en)
            result_A = self.subprotocols["teleport_A"].get_signal_result(Signals.SUCCESS, self)
            result_B = self.subprotocols["teleport_B"].get_signal_result(Signals.SUCCESS, self)
            result_tel = {
                "pos_A0": result_A["pos_A0"],
                "pos_A1": result_A["pos_A1"],
                "pos_B": result_B,
                "time": sim_time() - start_time
            }
            #print(result_tel)
            self.send_signal(Signals.SUCCESS, [result_en, result_tel])
            #print(f"Simulation {i}: Finish")

def sim_setup(node_a, node_b, num_runs, epsilon):
    fb_example = FilteringBennet(node_a, node_b, num_runs=num_runs, epsilon=epsilon)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result_en, result_tel = protocol.get_signal_result(Signals.SUCCESS)
        #print(result_en)
        # Record fidelity
        node_a.qmemory.discard(positions=[result_tel["pos_A0"]]) # 使用しているメモリを解放
        node_a.qmemory.discard(positions=[result_tel["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result_tel["pos_B"]])
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)
        pairs = result_en["pairs1"] + result_en["pairs2"]
        prob_fil = (result_en["runs_f1"] + result_en["runs_f2"]) / pairs
        prob_be = 1.0 / result_en["runs_b"]
        prob = prob_fil * prob_be
        #print(f"prob_fil= {prob_fil}, prob_be= {prob_be}, prob= {prob}")
        return {"fidelity": f2, "pairs": pairs, "probability": prob,
                "probability_filter":prob_fil, "probability_bennet": prob_be, "time": result_tel["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=fb_example,
                                     event_type=Signals.SUCCESS.value))
    return fb_example, dc

def network_setup(source_delay=1e5, source_fidelity_sq=1.0, damp_rate=100,
                          node_distance=200):
    network = Network("network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=6, fallback_to_nonphysical=True))
    state_sampler = StateSampler([ks.b00, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    node_a.add_subcomponent(QSource("QSource_A1", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_a.add_subcomponent(QSource("QSource_A2", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=6, fallback_to_nonphysical=True))

    conn_cchannel1 = DirectConnection("CChannelConn_AB", 
        ClassicalChannel("CChannel_A->B", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}), 
        ClassicalChannel("CChannel_B->A", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel1, label="filter1",
                           port_name_node1="cout_bob1", port_name_node2="cin_alice1")
    conn_cchannel2 = DirectConnection("CChannelConn_AB", 
        ClassicalChannel("CChannel_A->B", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}), 
        ClassicalChannel("CChannel_B->A", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel2, label="filter2",
                           port_name_node1="cout_bob2", port_name_node2="cin_alice2")    
    conn_cchannel3 = DirectConnection("CChannelConn_AB", 
        ClassicalChannel("CChannel_A->B", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}), 
        ClassicalChannel("CChannel_B->A", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel3, label="bemmet",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")    
    
    # quantum_noise_modelに振幅減衰ノイズを指定
    # "quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False)
    # "quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False)
    # "quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False)
    qchannel1 = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False),
                                      "delay_model": FibreDelayModel(c=200e3)})
    port_name_a1, port_name_b1 = network.add_connection(node_a, node_b, channel_to=qchannel1, label="quantum1",
                                                      port_name_node1="qout_bob1", port_name_node2="qin_alice1")
    qchannel2 = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False),
                                      "delay_model": FibreDelayModel(c=200e3)})
    port_name_a2, port_name_b2 = network.add_connection(node_a, node_b, channel_to=qchannel2, label="quantum2",
                                                      port_name_node1="qout_bob2", port_name_node2="qin_alice2")

    # Link Alice ports:
    node_a.subcomponents["QSource_A1"].ports["qout1"].forward_output(
        node_a.ports[port_name_a1])
    node_a.subcomponents["QSource_A1"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    node_a.subcomponents["QSource_A2"].ports["qout1"].forward_output(
        node_a.ports[port_name_a2])
    node_a.subcomponents["QSource_A2"].ports["qout0"].connect(
        node_a.qmemory.ports["qin1"])    
    # Link Bob ports:
    node_b.ports[port_name_b1].forward_input(node_b.qmemory.ports["qin0"])
    node_b.ports[port_name_b2].forward_input(node_b.qmemory.ports["qin1"])
    return network

def run_experiment(node_distances, variable):
    fidelity_data = pandas.DataFrame()
    for node_distance in node_distances:
        for epsilon in variable:
            ns.sim_reset()
            network = network_setup(node_distance=node_distance)
            node_a = network.get_node("node_A")
            node_b = network.get_node("node_B")
            pro_example, dc = sim_setup(node_a, node_b, 1000, epsilon)
            pro_example.start()
            ns.sim_run()
            df = dc.dataframe
            df['node_distance'] = node_distance
            df['epsilon'] = epsilon
            fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data

def save_heatmap(dataframe, value_col, title, colorbar_label, filename_prefix, node_distances, variable):
    data = dataframe.groupby(["node_distance", "epsilon"])[value_col].mean().reset_index()
    heatmap_data = data.pivot(index='epsilon', columns='node_distance', values=value_col)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, origin='lower', aspect='auto',
                    extent=[
                        min(node_distances), max(node_distances),
                        min(variable), max(variable)
                    ]
    )
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel('node_distance [km]')
    plt.ylabel(r'WM strength $\epsilon$')
    plt.title(title)
    save_dir = "./plots/filter-bennet/node_distance/phase"
    count1 = len([f for f in os.listdir(save_dir) if f.startswith(filename_prefix)])
    filename = f"{save_dir}/{filename_prefix}_{count1 + 1}.png"
    plt.savefig(filename)
    plt.close()
    if value_col == "fidelity":
        best_rows = data.loc[data.groupby("node_distance")["fidelity"].idxmax()]
        best_rows = best_rows.sort_values("node_distance")

        print("Optimal value for each node distance:")
        print(best_rows)
        count = len([f for f in os.listdir(save_dir)
                if f.startswith("optimal summary")])
        best_rows.to_csv(
            f"{save_dir}/optimal summary_{count + 1}.csv",
            index=False
        )

    print(f"Plot saved as {filename}")

def create_plot():
    matplotlib.use('Agg')
    node_distances = [i for i in range(10, 1000, 50)]
    variable = [i for i in np.arange(0.1, 1.0, 0.1)]
    datas = run_experiment(node_distances, variable)
    # 忠実度
    save_heatmap(
        datas,
        value_col="fidelity",
        title="Fidelity Heatmap with filtering & bennet\n(phase_damp_rate=100 Hz)",
        colorbar_label="Average Fidelity",
        filename_prefix="Filter-Bennet fidelity",
        node_distances=node_distances,
        variable=variable
    )
    # 成功確率
    save_heatmap(
        datas,
        value_col="probability",
        title="Success Probability Heatmap with filtering & bennet\n(phase_damp_rate=100 Hz)",
        colorbar_label="Average Success Probability",
        filename_prefix="Filter-Bennet probability",
        node_distances=node_distances,
        variable=variable
    )
    # エンタングルメントペア消費数
    save_heatmap(
        datas,
        value_col="pairs",
        title="Entanglement Pair Usage Heatmap with filtering & bennet\n(phase_damp_rate=100 Hz)",
        colorbar_label="Average Number of Pairs",
        filename_prefix="Filter-Bennet pairs",
        node_distances=node_distances,
        variable=variable
    )
    save_dir = "./plots/filter-bennet/node_distance/phase"
    count = len([f for f in os.listdir(save_dir) if f.startswith("Filter-Bennet result")])
    datas.to_csv(f"{save_dir}/Filter-Bennet result_{count + 1}.csv")

def run_experiment_noise(noise_rate, variable):
    fidelity_data = pandas.DataFrame()
    for noise in noise_rate:
        for epsilon in variable:
            ns.sim_reset()
            network = network_setup(damp_rate=noise)
            node_a = network.get_node("node_A")
            node_b = network.get_node("node_B")
            pro_example, dc = sim_setup(node_a, node_b, 1000, epsilon)
            pro_example.start()
            ns.sim_run()
            df = dc.dataframe
            df['damp_rate'] = noise
            df['epsilon'] = epsilon
            fidelity_data = pandas.concat([fidelity_data, df])
    return fidelity_data

def save_heatmap_noise(dataframe, value_col, title, colorbar_label, filename_prefix, noise_rate, variable):
    data = dataframe.groupby(["damp_rate", "epsilon"])[value_col].mean().reset_index()
    heatmap_data = data.pivot(index='epsilon', columns='damp_rate', values=value_col)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, origin='lower', aspect='auto',
                    extent=[
                        min(noise_rate), max(noise_rate),
                        min(variable), max(variable)
                    ]
    )
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel('phase_damp_rate [Hz]')
    plt.ylabel(r'WM strength $\epsilon$')
    plt.title(title)
    save_dir = "./plots/filter-bennet/noise/phase"
    count1 = len([f for f in os.listdir(save_dir) if f.startswith(filename_prefix)])
    filename = f"{save_dir}/{filename_prefix}_{count1 + 1}.png"
    plt.savefig(filename)
    plt.close()
    if value_col == "fidelity":
        best_rows = data.loc[data.groupby("damp_rate")["fidelity"].idxmax()]
        best_rows = best_rows.sort_values("damp_rate")

        print("Optimal value for each node distance:")
        print(best_rows)
        count = len([f for f in os.listdir(save_dir)
                if f.startswith("optimal summary")])
        best_rows.to_csv(
            f"{save_dir}/optimal summary_{count + 1}.csv",
            index=False
        )

    print(f"Plot saved as {filename}")

def create_plot_noise():
    matplotlib.use('Agg')
    noise_rate = [i for i in range(0, 1500, 100)]
    variable = [i for i in np.arange(0.1, 1.0, 0.1)]
    datas = run_experiment_noise(noise_rate, variable)
    # 忠実度
    save_heatmap_noise(
        datas,
        value_col="fidelity",
        title="Fidelity Heatmap with filtering & bennet\n(node_distance=200 km)",
        colorbar_label="Average Fidelity",
        filename_prefix="Filter-Bennet fidelity",
        noise_rate=noise_rate,
        variable=variable
    )
    # 成功確率
    save_heatmap_noise(
        datas,
        value_col="probability",
        title="Success Probability Heatmap with filtering & bennet\n(node_distance=200 km)",
        colorbar_label="Average Success Probability",
        filename_prefix="Filter-Bennet probability",
        noise_rate=noise_rate,
        variable=variable
    )
    # エンタングルメントペア消費数
    save_heatmap_noise(
        datas,
        value_col="pairs",
        title="Entanglement Pair Usage Heatmap with filtering & bennet\n(node_distance=200 km)",
        colorbar_label="Average Number of Pairs",
        filename_prefix="Filter-Bennet pairs",
        noise_rate=noise_rate,
        variable=variable
    )
    save_dir = "./plots/filter-bennet/noise/phase"
    count = len([f for f in os.listdir(save_dir) if f.startswith("Filter-Bennet result")])
    datas.to_csv(f"{save_dir}/Filter-Bennet result_{count + 1}.csv")

if __name__ == "__main__":
    #network = network_setup()
    #fb_example, dc = sim_setup(network.get_node("node_A"), network.get_node("node_B"), num_runs=2, epsilon=0.9)
    #fb_example.start()
    #ns.sim_run()
    #print("Average fidelity : {}".format(dc.dataframe["fidelity"].mean()))
    create_plot_noise()