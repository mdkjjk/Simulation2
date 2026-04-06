import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
from pydynaa import EventExpression

def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=1000,
                          node_distance=20):
    """Create an example network for use with the purification protocols.

    Returns
    -------
    :class:`~netsquid.components.component.Component`
        A network component with nodes and channels as subcomponents.

    Notes
    -----
        This network is also used by the matching integration test.

    """
    network = Network("purify_network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    state_sampler = StateSampler(
        [ks.b01, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    node_a.add_subcomponent(QSource(
        "QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    conn_cchannel = DirectConnection(
        "CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel)
    # node_A.connect_to(node_B, conn_cchannel)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_loss_model": None,
                                      "delay_model": FibreDelayModel(c=200e3)},
                              depolar_rate=0)
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

network = example_network_setup()