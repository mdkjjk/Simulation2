import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
import math
from matplotlib import pyplot as plt

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.qubitapi import fidelity, discard, multi_operate
from netsquid.qubits.ketstates import s00, b00, s0, s1, h0, h1, y0, y1
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
from netsquid.qubits.dmtools import DenseDMRepr
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, INSTR_H, INSTR_INIT, IGate
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

#時間非依存位相減衰ノイズ
def phase_dampen(qubit, prob=1):
    krausops1 = ops.Operator("E1", [[1, 0],[0, np.sqrt(1-prob)]])
    krausops2 = ops.Operator("E2", [[0, 0],[0, np.sqrt(prob)]])
    krausops = [krausops1, krausops2]

    multi_operate([qubit], krausops)

#時間依存位相減衰ノイズ
def delay_phase_dampen(qubit, gamma, delay):
    if gamma < 0:
        raise ValueError(f"damp_rate {gamma} should be non-negative.")
    
    damp_rate = 1 - np.exp(-2 * gamma * delay * 1e-9)
    phase_dampen(qubit, damp_rate)

#位相減衰ノイズモデル
class PhaseNoiseModel(QuantumErrorModel):
    def __init__(self, gamma, time_independent=False, **kwargs):
        super().__init__(**kwargs)
        # NOTE time independence should be set *before* the rate
        self.add_property('time_independent', time_independent, value_type=bool)

        def gamma_constraint(value):
            if self.time_independent and not 0 <= value <= 1:
                return False
            elif value < 0:
                return False
            return True
        self.add_property('gamma', gamma,
                          value_type=(int, float),
                          value_constraints=ValueConstraint(gamma_constraint))
    
    @property
    def gamma(self):
        return self.properties['gamma']
    
    @gamma.setter
    def gamma(self, value):
        self.properties['gamma'] = value

    @property
    def time_independent(self):
        """bool: Whether the probability of depolarizing is time independent."""
        return self.properties['time_independent']

    @time_independent.setter
    def time_independent(self, value):
        self.properties['time_independent'] = value
    
    def error_operation(self, qubits, delta_time=0, **kwargs):
        if self.time_independent:   # 時間非依存
            for qubit in qubits:
                if qubit is not None:
                    phase_dampen(qubit, self.gamma)
        else:                       # 時間依存
            for qubit in qubits:
                if qubit is not None:
                    delay_phase_dampen(qubit, gamma=self.gamma, delay=delta_time)

# Alice側のプロトコル
class Prepare(NodeProtocol):
    def __init__(self, node, port, start_expression=None, msg_header="wmeasure", name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "PrepareNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        self.start_expression = start_expression
        self.header = msg_header
        self._program = self._init_program()
        self._qmem_positions = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        
    def _init_program(self):
        prog = QuantumProgram(num_qubits=1)
        q1, = prog.get_qubit_indices(1)
        prog.apply(INSTR_INIT, [q1])
        prog.apply(INSTR_H, [q1])
        return prog

    def run(self):
        while True:
            yield self.start_expression
            yield self.node.qmemory.execute_program(self._program)
            self._qmem_positions = self.node.qmemory.used_positions
            qubit = self.node.qmemory.peek(positions=self._qmem_positions)
            state = np.array([[ns.qubits.reduced_dm(qubit)[0][0], ns.qubits.reduced_dm(qubit)[0][1]],
                              [ns.qubits.reduced_dm(qubit)[1][0], ns.qubits.reduced_dm(qubit)[1][1]]])
            #print(state)
            self.node.qmemory.pop(positions=self._qmem_positions)
            self._qmem_positions = None
            self.send_signal(Signals.SUCCESS, state)
    
    def _clear_qmem_positions(self):
        self.node.qmemory.pop(positions=self._qmem_positions)
        self._qmem_positions = None
    
    def start(self):
        return super().start()

# Bob側のプロトコル
class WMeasure(NodeProtocol):
    def __init__(self, node, port, start_expression=None, msg_header="wmeasure", theta=0.4, eta=0.6, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "WMeasureNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        self.start_expression = start_expression
        self.header = msg_header
        self._qmem_positions = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_wmeasurement_operators(theta)
        self._set_rotation_operators(eta)

    def _set_wmeasurement_operators(self, theta):
        mxp = ops.Operator("Mx+", np.cos(theta/2) * outerprod(h0) + np.sin(theta/2) * outerprod(h1))
        mxm = ops.Operator("Mx-", np.cos(theta/2) * outerprod(h1) + np.sin(theta/2) * outerprod(h0))
        myp = ops.Operator("My+", np.cos(theta/2) * outerprod(y0) + np.sin(theta/2) * outerprod(y1))
        mym = ops.Operator("My-", np.cos(theta/2) * outerprod(y1) + np.sin(theta/2) * outerprod(y0))
        mzp = ops.Operator("Mz+", np.cos(theta/2) * outerprod(s0) + np.sin(theta/2) * outerprod(s1))
        mzm = ops.Operator("Mz-", np.cos(theta/2) * outerprod(s1) + np.sin(theta/2) * outerprod(s0))
        self.meas_ops_x = [mxp, mxm]
        self.meas_ops_y = [myp, mym]
        self.meas_ops_z = [mzp, mzm]

    def _set_rotation_operators(self, eta):
        Rxp = ops.Operator("Rx+", [[np.cos(eta/2), -np.sin(eta/2)*1j],[np.sin(eta/2)*1j, np.cos(eta/2)]])
        Rxm = ops.Operator("Rx-", [[np.cos(eta/2), np.sin(eta/2)*1j],[-np.sin(eta/2)*1j, np.cos(eta/2)]])
        Ryp = ops.Operator("Ry+", [[np.cos(eta/2), -np.sin(eta/2)],[np.sin(eta/2), np.cos(eta/2)]])
        Rym = ops.Operator("Ry-", [[np.cos(eta/2), np.sin(eta/2)],[-np.sin(eta/2), np.cos(eta/2)]])
        Rzp = ops.Operator("Rz+", [[np.exp((eta/2)*1j), 0],[0, np.exp(-(eta/2)*1j)]])
        Rzm = ops.Operator("Rz-", [[np.exp(-(eta/2)*1j), 0],[0, np.exp((eta/2)*1j)]])
        self.rot_ops_x = [Rxp, Rxm]
        self.rot_ops_y = [Ryp, Rym]
        self.rot_ops_z = [Rzp, Rzm]

    def _rotation_program(self, mresult, rot_axis):
        INSTR_R = IGate("R_gate", operator=rot_axis[0]) if mresult == 0 else IGate("R_gate", operator=rot_axis[1])
        prog = QuantumProgram(num_qubits=1)
        q1 = prog.get_qubit_indices(1)
        prog.apply(INSTR_R, q1)
        return prog

    def run(self):
        while True:
            yield self.await_port_input(self.port)
            self._qmem_positions = self.node.qmemory.used_positions
            #print(self._qmem_positions)
            yield from self._handle_new_qubit(self._qmem_positions)
            self.send_signal(Signals.SUCCESS, self._qmem_positions[0])

    def _handle_new_qubit(self, memory_position):
        assert not self.node.qmemory.mem_positions[memory_position[0]].is_empty
        qubit = self.node.qmemory.peek(positions=memory_position)
        state = ns.qubits.reduced_dm(qubit)
        #print(state)
        ccs = [np.abs(2 * state[0][1].real), np.abs(2 * state[0][1].imag), np.abs(2 * state[0][0] - 1)]
        #print(ccs)
        axis = ccs.index(max(ccs))
        yield from self._weak_measurement(ccs, axis)

    def _weak_measurement(self, ccs, axis):
        if axis == 0:
            if ccs[1] > ccs[2]:
                meas_operators = self.meas_ops_y
                rot_axis = self.rot_ops_z
            else:
                meas_operators = self.meas_ops_z
                rot_axis = self.rot_ops_y
        elif axis == 1:
            if ccs[0] > ccs[2]:
                meas_operators = self.meas_ops_x
                rot_axis = self.rot_ops_z
            else:
                meas_operators = self.meas_ops_z
                rot_axis = self.rot_ops_x
        else:
            if ccs[0] > ccs[1]:
                meas_operators = self.meas_ops_x
                rot_axis = self.rot_ops_y
            else:
                meas_operators = self.meas_ops_y
                rot_axis = self.rot_ops_x
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, self._qmem_positions, meas_operators=meas_operators)
        mresult = output[0]["instr"][0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        yield self.node.qmemory.execute_program(self._rotation_program(mresult, rot_axis), self._qmem_positions)

def network_setup(source_delay=1e5, source_fidelity_sq=0.9, damp_rate=50, node_distance=2000):
    network = Network("wmeasure_network")

    # ノード設定
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=11,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    node_b.add_subcomponent(QuantumProcessor("QuantumMemory_B", num_positions=11,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能

    # チャネル設定
    conn_cchannel = DirectConnection("CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel,
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    # quantum_noise_modelに振幅減衰ノイズを指定
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": PhaseNoiseModel(gamma=damp_rate, time_independent=False),
                                      "delay_model": FibreDelayModel(c=200e3)})
    network.add_connection(node_a, node_b, channel_to=qchannel, label="quantum",
                           port_name_node1="qout_bob", port_name_node2="qin_alice")
    
    # Link Alice ports:
    node_a.qmemory.ports["qout"].forward_output(node_a.ports["qout_bob"])
    # Link Bob ports:
    node_b.ports["qin_alice"].forward_input(node_b.qmemory.ports["qin0"])
    return network

class WMeasureExample(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="WMeasure example")
        self.num_runs = num_runs

        self.add_subprotocol(Prepare(node_a, node_a.ports["qout_bob"], name="wmeasure_A"))
        self.add_subprotocol(WMeasure(node_b, node_b.ports["qin_alice"], theta=0.3, eta=0.5, name="wmeasure_B"))

        self.subprotocols["wmeasure_A"].start_expression = self.subprotocols["wmeasure_A"].await_signal(self, Signals.WAITING)
        self.subprotocols["wmeasure_B"].start_expression = self.subprotocols["wmeasure_B"].await_signal(self, Signals.WAITING)
    
    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["wmeasure_A"], Signals.SUCCESS) &
                        self.await_signal(self.subprotocols["wmeasure_B"], Signals.SUCCESS))
            signal_A = self.subprotocols["wmeasure_A"].get_signal_result(Signals.SUCCESS, self)
            signal_B = self.subprotocols["wmeasure_B"].get_signal_result(Signals.SUCCESS, self)
            result = {
                "ideal_state": signal_A,
                "pos_B": signal_B,
                "time": sim_time() - start_time
            }
            self.send_signal(Signals.SUCCESS, result)
        
def sim_setup(node_a, node_b, num_runs):
    wm_example = WMeasureExample(node_a, node_b, num_runs)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        #print(result)
        ideal_state = result["ideal_state"]
        q, = node_b.qmemory.pop(positions=[result["pos_B"]])
        f2 = qapi.fidelity(q, ideal_state, squared=True)
        return {"F2": f2, "time": result["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=wm_example,
                                     event_type=Signals.SUCCESS.value))
    return wm_example, dc

def run_experiment(variables):
    fidelity_data = pandas.DataFrame()
    

network = network_setup()
wm_example, dc = sim_setup(network.get_node("node_A"), network.get_node("node_B"), 1)
wm_example.start()
ns.sim_run()
print("Average fidelity of generated entanglement with WM: {}".format(dc.dataframe["F2"].mean()))