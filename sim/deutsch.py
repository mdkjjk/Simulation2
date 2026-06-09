#蒸留操作を加えたシミュレーション
import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
import noise
from noise import AmplitudeNoiseModel, PhaseNoiseModel

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


class Distil(NodeProtocol):
    """Protocol that does local DEJMPS distillation on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    role : "A" or "B"
        Distillation requires that one of the nodes ("B") conjugate its rotation,
        while the other doesn't ("A").
    start_expression : :class:`~pydynaa.EventExpression`
        EventExpression node should wait for before starting distillation.
        The EventExpression should have a protocol as source, this protocol should signal the quantum memory position
        of the qubit.
    msg_header : str, optional
        Value of header meta field used for classical communication.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """
    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self._program = self._setup_dejmp_program(conj_rotation)
        # self.INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
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

    def _setup_dejmp_program(self, conj_rotation):
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1])
        prog.apply(INSTR_ROT, [q2])
        prog.apply(INSTR_CNOT, [q1, q2])
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False)
        return prog

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            #print(f"{self.name}: Start")
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    #print(f"{self.name}: Result received at {classical_message} / time: {sim_time()}")
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                #print(f"{self.name}: Entanglement received at {ready_signal.result} / time: {sim_time()}")
                yield from self._handle_new_qubit(ready_signal.result)
            self._check_success()

    def start(self):
        # Clear any held qubits
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        #print(f"{self.name}: pop_positions = {positions} at _clear_qmem_positions")
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]
        #print(f"{self.name}: qmem_positions = {self._qmem_positions}")

    def _handle_new_qubit(self, memory_position):
        # Process signalling of new entangled qubit
        assert not self.node.qmemory.mem_positions[memory_position].is_empty
        if self._waiting_on_second_qubit:
            # Second qubit arrived: perform distil
            assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
            assert memory_position != self._qmem_positions[0]
            self._qmem_positions[1] = memory_position
            #print(f"{self.name}: qmem_positions = {self._qmem_positions}")
            self._waiting_on_second_qubit = False
            self.num_runs += 1
            #print(f"{self.name}: Sim {self.num_runs}")
            yield from self._node_do_DEJMPS()
        else:
            # New candidate for first qubit arrived
            # Pop previous qubit if present:
            pop_positions = [p for p in self._qmem_positions if p is not None and p != memory_position]
            #print(f"{self.name}: pop_positions = {pop_positions} at _handle_new_qubit")
            if len(pop_positions) > 0:
                self.node.qmemory.pop(positions=pop_positions)
            # Set new position:
            self._qmem_positions[0] = memory_position
            self._qmem_positions[1] = None
            #print(f"{self.name}: qmem_positions = {self._qmem_positions}")
            self.local_qcount += 1
            self.local_meas_result = None
            self._waiting_on_second_qubit = True

    def _node_do_DEJMPS(self):
        # Perform DEJMPS distillation protocol locally on one node
        #print(f"{self.name}: qmem_positions = {self._qmem_positions}")
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # We perform local DEJMPS
        yield self.node.qmemory.execute_program(self._program, [pos1, pos2])  # If instruction not instant
        self.local_meas_result = self._program.output["m"][0]
        self._qmem_positions[1] = None
        #print(f"{self.name}: qmem_positions = {self._qmem_positions}")
        # Send local results to the remote node to allow it to check for success.
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))

    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, [self._qmem_positions[0], self.num_runs])
                #print(f"{self.name}: SUCCESS / time: {sim_time()}")
                self.num_runs = 0
            else:
                # FAILURE
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
                #print(f"{self.name}: FAIL / time: {sim_time()}")
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


class InitStateProgram(QuantumProgram):
    default_num_qubits = 1

    def program(self):
        q1, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_S, q1)
        yield self.run()


class BellMeasurement(NodeProtocol):
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
            #print(f"{self.name}: Start")
            expr_port = self.start_expression
            yield expr_port
            entanglement_ready = True
            source_protocol = expr_port.atomic_source
            ready_signal = source_protocol.get_signal_by_event(event=expr_port.triggered_events[0], receiver=self)
            self._qmem_pos1 = ready_signal.result[0]
            #print(f"{self.name}: Entanglement received at {self._qmem_pos1} / time: {sim_time()}")
            self._qmem_pos0 = self.node.qmemory.unused_positions[0]
            self.node.qmemory.execute_program(qubit_init_program, qubit_mapping=[self._qmem_pos0])
            expr_signal = self.await_program(self.node.qmemory)
            yield expr_signal
            qubit_initialised = True
            #print(f"{self.name}: Initqubit received at {self._qmem_pos0} / time: {sim_time()}")
            if qubit_initialised and entanglement_ready:
                self.node.qmemory.operate(ns.CNOT, [self._qmem_pos0, self._qmem_pos1])
                self.node.qmemory.operate(ns.H, self._qmem_pos0)
                m, _ = self.node.qmemory.measure([self._qmem_pos0, self._qmem_pos1])
                # Send measurement results to Bob:
                self.port.tx_output(m)
                result = {"pos_A0": self._qmem_pos0,
                          "pos_A1": self._qmem_pos1,}
                self.send_signal(Signals.SUCCESS, result)
                #print(f"{self.name}: Finish / time: {sim_time()}")
                qubit_initialised = False
                entanglement_ready = False


class Correction(NodeProtocol):
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
            expr = yield (self.await_port_input(port_alice) | expr_signal)
            if expr.first_term.value:
                meas_results = port_alice.rx_input().items
                #print(f"{self.name}: Result: {meas_results} / time: {sim_time()}")
            else:
                entanglement_ready = True
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(event=expr.second_term.triggered_events[-1], receiver=self)
                self._qmem_pos = ready_signal.result[0]
                #print(f"{self.name}: Entanglement received at {self._qmem_pos} / time: {sim_time()}")
            if meas_results is not None and entanglement_ready:
                # Do corrections (blocking)
                if meas_results[0] == 1:
                    self.node.qmemory.execute_instruction(instr.INSTR_Z, [self._qmem_pos])
                if meas_results[1] == 1:
                    self.node.qmemory.execute_instruction(instr.INSTR_X, [self._qmem_pos])
                self.send_signal(Signals.SUCCESS, self._qmem_pos)
                #print(f"{self.name}: Teleport success / time: {sim_time()}")
                entanglement_ready = False
                meas_results = None


class DistilExample(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Distil example")
        self.num_runs = num_runs
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=2, name="entangle_A"))
        self.add_subprotocol(
            EntangleNodes(node=node_b, role="receiver", input_mem_pos=0, num_pairs=2,
                          name="entangle_B"))
        self.add_subprotocol(Distil(node_a, node_a.ports["cout_bob_dis"], role="A", name="purify_A"))
        self.add_subprotocol(Distil(node_b, node_b.ports["cin_alice_dis"], role="B", name="purify_B"))
        self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
        self.add_subprotocol(Correction(node=node_b, name="teleport_B"))
        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(self.subprotocols["purify_A"], Signals.SUCCESS)
        self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(self.subprotocols["purify_B"], Signals.SUCCESS)

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i}: Start")
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS))
            signal_A_pur = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result_pur = {"runs": signal_A_pur[1]}
            signal_A_tel = self.subprotocols["teleport_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_B_tel = self.subprotocols["teleport_B"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result_tel = {
                "pos_A0": signal_A_tel["pos_A0"],
                "pos_A1": signal_A_tel["pos_A1"],
                "pos_B": signal_B_tel,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, [result_tel, result_pur])
            #print(f"Simulation {i}: Finish")


def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=1500,
                          node_distance=30):
    network = Network("network")

    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemory_A", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0)))
    state_sampler = StateSampler([ks.b00, ks.s00],
        probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    node_a.add_subcomponent(QSource(
        "QSource_A", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemory_B", num_positions=2, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(0)))
    node_a.add_ports(["cout_bob_dis", "cout_bob_fil"])
    node_b.add_ports(["cin_alice_dis", "cin_alice_fil"])

    conn_cchannel_dis = DirectConnection("CChannelConn_dis_AB",
        ClassicalChannel("CChannel_dis_A->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_dis_B->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel_dis, label="distil",
                           port_name_node1="cout_bob_dis", port_name_node2="cin_alice_dis")
    conn_cchannel_fil = DirectConnection("CChannelConn_fil_AB",
        ClassicalChannel("CChannel_fil_A->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_fil_B->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel_fil, label="filter",
                           port_name_node1="cout_bob_fil", port_name_node2="cin_alice_fil")
    cchannel = DirectConnection("CChannelConn_tel", ClassicalChannel("CChannel_dis_A->B", length=node_distance,
                                models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=cchannel, label="tereport",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    # node_A.connect_to(node_B, conn_cchannel)
    qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                              models={"quantum_noise_model": DepolarNoiseModel(depolar_rate),
                                      "delay_model": FibreDelayModel(c=200e3)},
                              depolar_rate=0)
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charlie")

    # Link Alice ports:
    node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSource_A"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    # Link Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.qmemory.ports["qin0"])
    return network


def example_sim_setup(node_a, node_b, num_runs):
    """Example simulation setup for purification protocols.

    Returns
    -------
    :class:`~netsquid.examples.purify.FilteringExample`
        Example protocol to run.
    :class:`pandas.DataFrame`
        Dataframe of collected data.

    """
    filt_example = DistilExample(node_a, node_b, num_runs=num_runs)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        #print(result)
        # Record fidelity
        node_a.qmemory.pop(positions=[result[0]["pos_A0"]])
        node_a.qmemory.pop(positions=[result[0]["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result[0]["pos_B"]])
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)
        prob = 1 / result[1]["runs"]
        return {"F2": f2, "pairs": result[0]["pairs"], "probability": prob, "time": result[0]["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=filt_example,
                                     event_type=Signals.SUCCESS.value))
    return filt_example, dc


def run_experiment(node_distances):
    fidelity_data = pandas.DataFrame()
    for node_distance in node_distances:
        ns.sim_reset()
        network = example_network_setup(node_distance=node_distance)
        node_a = network.get_node("node_A")
        node_b = network.get_node("node_B")
        example, dc = example_sim_setup(node_a, node_b, 100)
        example.start()
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
        title="Fidelity of the teleported quantum state with deutsch",
        prefix="Deutsch fidelity"
    )
    save_plot(
        datas,
        column="probability",
        title="Probability of success - deutsch",
        prefix="Deutsch probability"
    )
    save_plot(
        datas,
        column="pairs",
        title="Number of entanglement pairs used with deutsch",
        prefix="Deutsch pairs"
    )
    save_dir = "./plots_test"
    count = len([f for f in os.listdir(save_dir) if f.startswith("Deutsch result")])
    datas.to_csv(f"{save_dir}/Deutsch result_{count + 1}.csv")


if __name__ == "__main__":
    #network = example_network_setup()
    #filt_example, dc = example_sim_setup(network.get_node("node_A"),network.get_node("node_B"),num_runs=5)
    #filt_example.start()
    #ns.sim_run()
    #print("Average fidelity of generated entanglement with distil: {}".format(dc.dataframe["F2"].mean()))
    #print("Average resource with protection: {}".format(dc.dataframe["pairs"].mean()))
    #print("Average probability of success with protection: {}".format(dc.dataframe["probability"].mean()))
    create_plot()