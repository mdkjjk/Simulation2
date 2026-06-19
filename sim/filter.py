import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
import noise
from noise import AmplitudeNoiseModel, PhaseNoiseModel
import teleportation
from teleportation import InitStateProgram, BellMeasurement, Correction

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
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.examples.entanglenodes import EntangleNodes
from pydynaa import EventExpression


class Filter(NodeProtocol):
    """Protocol that does local filtering on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    start_expression : :class:`~pydynaa.core.EventExpression` or None, optional
        Event expression node should wait for before starting filter.
        This event expression should have a
        :class:`~netsquid.protocols.protocol.Protocol` as source and should by fired
        by signalling a signal by this protocol, with the position of the qubit on the
        quantum memory as signal result.
        Must be set before the protocol can start
    msg_header : str, optional
        Value of header meta field used for classical communication.
    epsilon : float, optional
        Parameter used in filter's measurement operator.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    Attributes
    ----------
    meas_ops : list
        Measurement operators to use for filter general measurement.

    """

    def __init__(self, node, port, start_expression=None, msg_header="filter",
                 epsilon=0.3, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "Filter({}, {})".format(node.name, port.name)
        super().__init__(node, name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.num_runs = 0
        self.local_qcount = 0
        self.local_meas_OK = False
        self.remote_qcount = 0
        self.remote_meas_OK = False
        self.header = msg_header
        self._qmem_pos = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_measurement_operators(epsilon)

    def _set_measurement_operators(self, epsilon):
        m0 = ops.Operator("M0", np.sqrt(epsilon) * outerprod(s0) + outerprod(s1))
        m1 = ops.Operator("M1", np.sqrt(1 - epsilon) * outerprod(s0))
        self.meas_ops = [m0, m1]

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_OK = classical_message.items
                    #print(f"{self.name}: Result received at {classical_message} / time: {sim_time()}")
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                self._qmem_pos = ready_signal.result
                #print(f"{self.name}: Entanglement received at {self._qmem_pos} / time: {sim_time()}")
                yield from self._handle_qubit_rx()

    # TODO does start reset vars?
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_OK = False
        self.remote_meas_OK = False
        return super().start()

    def stop(self):
        super().stop()
        # TODO should stop clear qmem_pos?
        if self._qmem_pos and self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    def _handle_qubit_rx(self):
        # Handle incoming Qubit on this node.
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        self.num_runs += 1
        #print(f"{self.name}: Sim {self.num_runs}")
        # Retrieve Qubit from input store
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        m = output["instr"][0]
        # m = INSTR_MEASURE(self.node.qmemory, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        self.local_qcount += 1
        self.local_meas_OK = (m == 0)
        self.port.tx_output(Message([self.local_qcount, self.local_meas_OK], header=self.header))
        self._check_success()

    def _handle_cchannel_rx(self):
        # Handle incoming classical message from sister node.
        if (self.local_qcount == self.remote_qcount and
                self._qmem_pos is not None and
                self.node.qmemory.mem_positions[self._qmem_pos].in_use):
            self._check_success()

    def _check_success(self):
        # Check if protocol succeeded after receiving new input (qubit or classical information).
        # Returns true if protocol has succeeded on this node
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_OK and self.remote_meas_OK):
            # SUCCESS!
            self.send_signal(Signals.SUCCESS, [self._qmem_pos, self.num_runs])
            #print(f"{self.name}: SUCCESS / time: {sim_time()}")
            self.num_runs = 0
        elif self.local_meas_OK and self.local_qcount > self.remote_qcount:
            # Need to wait for latest remote status
            pass
        else:
            # FAILURE
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)
            #print(f"{self.name}: FAIL / time: {sim_time()}")

    def _handle_fail(self):
        if self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 1:
            return False
        return True

class FilteringExample(LocalProtocol):
    r"""Protocol for a complete filtering experiment.

    Combines the sub-protocols:
    - :py:class:`~netsquid.examples.entanglenodes.EntangleNodes`
    - :py:class:`~netsquid.examples.purify.Filter`

    Will run for specified number of times then stop, recording results after each run.

    Parameters
    ----------
    node_a : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    node_b : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    num_runs : int
        Number of successful runs to do.
    epsilon : float
        Parameter used in filter's measurement operator.

    Attributes
    ----------
    results : :py:obj:`dict`
        Dictionary containing results. Results are :py:class:`numpy.array`\s.
        Results keys are *fidelity*, *pairs*, and *time*.

    Subprotocols
    ------------
    entangle_A : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node A.
    entangle_B : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node B.
    purify_A : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node A.
    purify_B : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node B.

    Notes
    -----
        The filter purification does not support the stabilizer formalism.

    """

    def __init__(self, node_a, node_b, num_runs, epsilon=0.9):
        super().__init__(nodes={"A": node_a, "B": node_b}, name="Filtering example")
        self._epsilon = epsilon
        self.num_runs = num_runs
        # Initialise sub-protocols
        self.add_subprotocol(EntangleNodes(node=node_a, role="source", input_mem_pos=0,
                                           num_pairs=1, name="entangle_A"))
        self.add_subprotocol(
            EntangleNodes(node=node_b, role="receiver", input_mem_pos=0, num_pairs=1,
                          name="entangle_B"))
        self.add_subprotocol(Filter(node_a, node_a.ports["cout_bob_fil"],
                                    epsilon=epsilon, name="purify_A1"))
        self.add_subprotocol(Filter(node_b, node_b.ports["cin_alice_fil"],
                                    epsilon=epsilon, name="purify_B1"))
        self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
        self.add_subprotocol(Correction(node=node_b, name="teleport_B"))
        # Set start expressions
        self.subprotocols["purify_A1"].start_expression = (
            self.subprotocols["purify_A1"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B1"].start_expression = (
            self.subprotocols["purify_B1"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
                            self.subprotocols["purify_A1"], Signals.FAIL) |
                            self.subprotocols["entangle_A"].await_signal(
                                self, Signals.WAITING))
        self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(self.subprotocols["purify_A1"], Signals.SUCCESS)
        self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(self.subprotocols["purify_B1"], Signals.SUCCESS)

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            #print(f"Simulation {i}: Start")
            start_time = sim_time()
            self.subprotocols["entangle_A"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS))
            signal_A_pur = self.subprotocols["purify_A1"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result_pur = {"runs": signal_A_pur[1]}
            signal_A = self.subprotocols["teleport_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_B = self.subprotocols["teleport_B"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result = {
                "pos_A0": signal_A["pos_A0"],
                "pos_A1": signal_A["pos_A1"],
                "pos_B": signal_B,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_A"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, [result, result_pur])
            #print(f"Simulation {i}: Finish")


def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=100,
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
                              models={"quantum_noise_model": AmplitudeNoiseModel(gamma=depolar_rate),
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
    filt_example = FilteringExample(node_a, node_b, num_runs=num_runs, epsilon=0.9)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        node_a.qmemory.pop(positions=[result[0]["pos_A0"]])
        node_a.qmemory.pop(positions=[result[0]["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result[0]["pos_B"]])
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)
        prob = 1 / result[1]["runs"]
        #print(f"{result["time"]}: pairs = {result["pairs"]}, fidelity = {fidelity}")
        return {"fidelity": f2, "pairs": result[0]["pairs"], "probability": prob, "time": result[0]["time"]}

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
        **{column:'mean', 'sem':'sem'}).reset_index()
    save_dir = "./plots_test/filter/node_distance"
    count1 = len([f for f in os.listdir(save_dir)
                 if f.startswith(prefix)])
    filename = f"{save_dir}/{prefix}_{count1 + 1}.png"
    data.plot(
        x='node_distance',
        y=column,
        yerr='sem',
        **plot_style
    )
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")
    count2 = len([f for f in os.listdir(save_dir)
                if f.startswith(column + " summary")])
    data[['node_distance', column]].to_csv(f"{save_dir}/{column} summary_{count2 + 1}.csv")

def create_plot():
    matplotlib.use('Agg')
    node_distances = [1 + i for i in range(0, 1000, 50)]
    #noise_rate = [i for i in range(0, 1500, 100)]
    datas = run_experiment(node_distances)
    save_plot(
        datas,
        column="fidelity",
        title="Fidelity of the teleported quantum state with filter - depolar_rate=100 Hz",
        prefix="Filter fidelity"
    )
    save_plot(
        datas,
        column="probability",
        title="Probability of success with filter - depolar_rate=100 Hz",
        prefix="Filter probability"
    )
    save_plot(
        datas,
        column="pairs",
        title="Number of entanglement pairs used with filter - depolar_rate=100 Hz",
        prefix="Filter pairs"
    )
    save_dir = "./plots_test/filter/node_distance"
    count = len([f for f in os.listdir(save_dir) if f.startswith("Filter result")])
    datas.to_csv(f"{save_dir}/Filter result_{count + 1}.csv")


if __name__ == "__main__":
    #network = example_network_setup()
    #filt_example, dc = example_sim_setup(network.get_node("node_A"),network.get_node("node_B"),num_runs=5)
    #filt_example.start()
    #ns.sim_run()
    #print("Average fidelity of generated entanglement with filtering: {}".format(dc.dataframe["fidelity"].mean()))
    #print("Average resource with protection: {}".format(dc.dataframe["pairs"].mean()))
    #print("Average probability of success with protection: {}".format(dc.dataframe["probability"].mean()))
    create_plot()