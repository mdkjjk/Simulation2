import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
from netsquid.qubits.dmtools import DenseDMRepr
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
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
    """ノード上で繰り返し法を行うプロトコル

    Parameters
    ----------


    """


def network_setup(source_delay=1e5, node_distance=100, depolar_rate=1000):
    network = Network("bennet_network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=11,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    state = DenseDMRepr(np.array[[(1-f)/3, 0, 0, 0], [0, (2*f+1)/6, (1-4*f)/6, 0],
                                 [0, (1-4*f)/6, (2*f+1)/6, 0], [0, 0, 0, (1-f)/3]])
    state_sampler = StateSampler