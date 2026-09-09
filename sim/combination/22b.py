# Protect + Non-breeding(Bennet)
import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
import math
from matplotlib import pyplot as plt
from noise import AmplitudeNoiseModel, PhaseNoiseModel
from teleportation import InitStateProgram, BellMeasurement, Correction
from bennet import Bennet
from wm2022 import LocalEntangle, Protect, RWMeasure

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

def network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=100, node_distance=10):
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
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=False),
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

class ProtectBennet(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs, omega, theta):
            super().__init__(nodes={"A": node_a, "B": node_b}, name="Protect example")
            self.num_runs = num_runs
            # エンタングルメント生成プロトコル
            self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A", input_mem_pos0=0,
                                               input_mem_pos1=1, num_pairs=1, name="entangle_A"))
            # 保護処理プロトコル
            self.add_subprotocol(Protect(node_a, node_a.ports["cout_bob"], omega=omega, name="protect_A"))
            self.add_subprotocol(RWMeasure(node_b, node_b.ports["cin_alice"],
                                 node_b.ports["qin_alice"], theta=theta, name="rwmeasure_B"))

            # 精製処理プロトコル
            self.add_subprotocol(Bennet(node_a, node_a.ports["cout_bob"], role="A", name="bennet_A"))
            self.add_subprotocol(Bennet(node_b, node_b.ports["cin_alice"], role="B", name="bennet_B"))

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

            # 精製処理プロトコルの開始条件                        
            self.subprotocols["bennet_A"].start_expression = (
                self.subprotocols["bennet_A"].await_signal(self.subprotocols["protect_A"],
                                                            Signals.SUCCESS))
            self.subprotocols["bennet_B"].start_expression = (
                self.subprotocols["bennet_B"].await_signal(self.subprotocols["rwmeasure_B"],
                                                            Signals.SUCCESS))

            # テレポーテーションプロトコルの開始条件
            self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(
                                                                self.subprotocols["protect_A"], Signals.SUCCESS)
            self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(
                                                                self.subprotocols["rwmeasure_B"], Signals.SUCCESS)