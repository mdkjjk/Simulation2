import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt

import netsquid.components.instructions as instr
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.examples.entanglenodes import EntangleNodes
from pydynaa import EventExpression
from netsquid.components import QuantumMemory
from netsquid.qubits.qubitapi import create_qubits

depolar_noise = DepolarNoiseModel(depolar_rate=1e6)
qmem = QuantumMemory("DepolarMemory", num_positions=2,
    memory_noise_models=[depolar_noise, depolar_noise])
for mem_pos in qmem.mem_positions:
    mem_pos.models['noise_model'] = depolar_noise
qubits = create_qubits(2)
qmem.put(qubits)
#qmem.put(q2)
print(qmem.peek(0))
print(qmem.peek(1))
for i in range(0, 2, 1):
    if qmem.mem_positions[i].in_use:
        qmem.pop(positions=[i])
#print(qmem.pop(positions=[0,1]))
print(qmem.peek(0))
print(qmem.peek(1))