import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
from noise import AmplitudeNoiseModel, PhaseNoiseModel
from teleportation import InitStateProgram, BellMeasurement, Correction
from filter import Filter
from wmb import Bennet
from entanglenodes import EntangleNodes

import netsquid.components.instructions as instr
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from pydynaa import EventExpression

ns.set_qstate_formalism(QFormalism.DM)

class FilteringBennet(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs, epsilon=0.9):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Filtering example")
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
            self.subprotocols["bennet_B'"].await_signal(self.subprotocols["filter_B1"], Signals.SUCCESS) &
            self.subprotocols["bennet_B'"].await_signal(self.subprotocols["filter_B2"], Signals.SUCCESS))         
                
        
    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i}: Start")
            start_time = sim_time()
            self.subprotocols["entangle_A1"].entangled_pairs = 0
            self.subprotocols["entangle_A2"].entangled_pairs = 0            
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["bennet_A'"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["bennet_B'"], Signals.SUCCESS))
            signal_A = self.subprotocols["bennet_A'"].get_signal_result(Signals.SUCCESS,self)
            signal_B = self.subprotocols["bennet_B'"].get_signal_result(Signals.SUCCESS,self)
                                                                       
            result = {
                "pos_A": signal_A[0],
                "pos_B": signal_B[0],               
                "time": sim_time() - start_time,
                "pairs1": self.subprotocols["entangle_A1"].entangled_pairs,
                "pairs2": self.subprotocols["entangle_A2"].entangled_pairs,
            }
            print(result)
            self.send_signal(Signals.SUCCESS, result)
            #print(f"Simulation {i}: Finish")

def sim_setup(node_a, node_b, num_runs, epsilon):
    fil_example = FilteringBennet(node_a, node_b, num_runs=num_runs, epsilon=epsilon)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity([q_A, q_B], ks.b00, squared=True)
        prob = 1 / result["runs"]
        #print(f"{result["time"]}: pairs = {result["pairs"]}, fidelity = {fidelity}")
        return {"fidelity": f2, "pairs": result["pairs"], "probability": prob, "time": result["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=fil_example,
                                     event_type=Signals.SUCCESS.value))
    return fil_example, dc

def network_setup(source_delay=1e5, source_fidelity_sq=0.8, damp_rate=500,
                          node_distance=200):
    network = Network("network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=6, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0)))
    state_sampler = StateSampler([ks.b00, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    node_a.add_subcomponent(QSource("QSource_A1", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_a.add_subcomponent(QSource("QSource_A2", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=6, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0)))

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
    
    
    # node_A.connect_to(node_B, conn_cchannel)
    # quantum_noise_modelに振幅減衰ノイズを指定
    # "quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False)
    # "quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False)
    # "quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False)
    qchannel1 = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False),
                                      "delay_model": FibreDelayModel(c=200e3)})
    port_name_a1, port_name_b1 = network.add_connection(node_a, node_b, channel_to=qchannel1, label="quantum1",
                                                      port_name_node1="qout_bob1", port_name_node2="qin_alice1")
    qchannel2 = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": AmplitudeNoiseModel(gamma=damp_rate, time_independent=False),
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

if __name__ == "__main__":
    network = network_setup()
    fil_example = FilteringBennet(network.get_node("node_A"), network.get_node("node_B"), num_runs=1, epsilon=0.9)
    fil_example.start()
    ns.sim_run()
    #print("Average fidelity of generated entanglement with filtering: {}".format(dc.dataframe["fidelity"].mean()))