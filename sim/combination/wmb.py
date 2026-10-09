import numpy as np
import netsquid as ns
import pydynaa as pd
import pandas
import matplotlib, os
from matplotlib import pyplot as plt
from noise import AmplitudeNoiseModel, PhaseNoiseModel
from teleportation import InitStateProgram, BellMeasurement, Correction
from wm2022 import LocalEntangle
from bennet import Bennet

from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from netsquid.qubits.qubitapi import fidelity, discard, combine_qubits
from netsquid.qubits.ketstates import s00, b00, y0
from netsquid.qubits.state_sampler import StateSampler
from netsquid.qubits.qformalism import QFormalism
from netsquid.qubits.dmtools import DenseDMRepr
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, INSTR_X, INSTR_Y, INSTR_SWAP
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
from pydynaa import EventExpression

ns.set_qstate_formalism(QFormalism.DM)

class Protect(NodeProtocol):   # Alice側のプロトコル
    def __init__(self, node, port, start_expression=None, msg_header="protect", omega=np.pi/3, pair_id=None, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "ProtectNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.pair_id = pair_id
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_positions = [None, None]
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_wmeasurement_operators(omega)

    def _set_wmeasurement_operators(self, omega):
        m0 = ops.Operator("M0", [[np.cos(omega/2), 0], [0, np.sin(omega/2)]])
        m1 = ops.Operator("M1", [[np.sin(omega/2), 0], [0, np.cos(omega/2)]])
        self.wmeas_ops = [m0, m1]

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            expr = yield cchannel_ready | qmemory_ready
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    #print(f"{self.name}: Bob's result received {classical_message}")
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                        event=expr.second_term.triggered_events[0], receiver=self) # エンタングルメントが保存されたメモリポジションを取得
                #print(f"{self.name}: Entanglement received at {ready_signal.result} / time: {sim_time()}")      
                self._qmem_positions[0] = ready_signal.result["mem_pos0"]
                self._qmem_positions[1] = ready_signal.result["mem_pos1"]
                yield from self._handle_qubit_rx()

    def start(self):
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        return super().start()

    def _handle_qubit_rx(self):
        #print(f"{self.name}: Sim {self.num_runs}")
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [pos2], 
                                                       meas_operators=self.wmeas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        self.local_meas_result = output["instr"][0]
        #print(f"{self.name}: Result = {self.local_meas_result}")
        self.local_qcount += 1
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result], header=self.header))
        #print(f"{self.name}: Sending classical result [{self.local_qcount}, {self.local_meas_result}]")
        if self.local_meas_result == 1:
            #print(f"{self.name}: Flip operation")
            if self.node.qmemory.busy:
                yield self.await_program(self.node.qmemory)
            self.node.qmemory.execute_instruction(INSTR_X, [pos2])
        meta_data = {"pair_id": self.pair_id,
                     "protocol": self.name,
                     "mem_pos": pos2}
        self.node.qmemory.pop(positions=pos2, meta_data=meta_data)
        #print(f"{self.name}: {self.node.qmemory.used_positions}")
        self._qmem_positions[1] = None

    def _handle_cchannel_rx(self):
        if (self._qmem_positions is not None and
                self.node.qmemory.mem_positions[self._qmem_positions[0]].in_use):
            self._check_success()

    def _check_success(self):
        #print(f"{self.name}: Remote result is {self.remote_meas_result}")
        if self.remote_meas_result == 1:
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)
            #print(f"{self.name}: FAIL")
            self.local_meas_result = None
            self.remote_meas_result = None
        else:
            self.send_signal(Signals.SUCCESS, [self._qmem_positions[0], self.local_qcount])
            #print(f"{self.name}: SUCCESS")
            self.local_qcount = 0

    def _handle_fail(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]

class RWMeasure(NodeProtocol):   # Bob側のプロトコル
    def __init__(self, node, port_c, dispatcher, pair_id, start_expression=None, msg_header="protect", theta=0.2, name=None):
        if not isinstance(port_c, Port):
            raise ValueError("{} is not a Port".format(Port))
        name = name if name else "RWMeasureNode({}, {})".format(node.name, Port.name)
        super().__init__(node, name=name)
        self.port_c = port_c
        self.dispatcher = dispatcher
        self.pair_id = pair_id
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_pos = None
        self.received_metadata = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_rwmeasurement_operators(theta)

    def _set_rwmeasurement_operators(self, theta):
        n0 = ops.Operator("N0", [[theta, 0], [0, 1]])
        n0_ = ops.Operator("N0_", [[np.sqrt(1-theta*theta), 0], [0, 0]])
        n1 = ops.Operator("N1", [[1, 0], [0, theta]])
        n1_ = ops.Operator("N1_", [[0, 0], [0, np.sqrt(1-theta*theta)]])
        self.rwmeas_ops0 = [n0, n0_]
        self.rwmeas_ops1 = [n1, n1_]
    
    def run(self):
        #print(f"{self.name}: RUN STARTED")

        while True:
            #print(f"{self.name}: waiting for classical or quantum message")
            cchannel_ready = self.await_port_input(self.port_c)
            dispatcher_ready = self.await_signal(self.dispatcher, Signals.SUCCESS)

            expr = yield cchannel_ready | dispatcher_ready

            # ==================================================
            # Aliceから古典測定結果
            # ==================================================

            if expr.first_term.value:
                #print(f"{self.name}: CLASSICAL CHANNEL EVENT")
                classical_message = self.port_c.rx_input()

                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    #print(f"{self.name}: Alice's result received {classical_message}")

            # ==================================================
            # Dispatcherから量子ビット到着通知
            # ==================================================

            elif expr.second_term.value:
                #print(f"{self.name}: DISPATCHER EVENT")
                signal_result = self.dispatcher.get_signal_result(Signals.SUCCESS, self)
                #print(f"{self.name}: Dispatcher signal = {signal_result}")

                pair_id = signal_result["pair_id"]
                bob_mem_pos = signal_result["bob_mem_pos"]

                # 自分が担当するpair_idか確認
                if pair_id != self.pair_id:
                    #print(f"{self.name}: Ignored pair_id={pair_id}")
                    continue

                self._qmem_pos = [bob_mem_pos]
                #print(f"{self.name}: Entanglement arrived at Bob qmemory position {self._qmem_pos}")

                # Alice側の測定結果を受信済みなら
                # weak measurementを実行
                if self.remote_meas_result is not None:
                    yield from self._handle_qubit_rx()
    
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_result = None
        self.remote_meas_result = None
        return super().start()

    def stop(self):
        super().stop()

    def _handle_qubit_rx(self):
        self.local_qcount += 1
        #print(f"{self.name}: Local qcount is {self.local_qcount}")
        pos = self._qmem_pos[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        if self.remote_meas_result == 1:
            #print(f"{self.name}: Remote result = 1 -> Flip operation")
            self.node.qmemory.execute_instruction(INSTR_X, [pos])
            if self.node.qmemory.busy:
                yield self.await_program(self.node.qmemory)
            output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [pos], meas_operators=self.rwmeas_ops1)
            self.local_meas_result = output[0]["instr"][0]
            #print(f"{self.name}: Result = {output}")
        else:
            #print(f"{self.name}: Remote result = 0")
            output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [pos], meas_operators=self.rwmeas_ops0)[0]
            self.local_meas_result = output["instr"][0]
            #print(f"{self.name}: Result = {self.local_meas_result}")
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        self.port_c.tx_output(Message([self.local_qcount, self.local_meas_result], header=self.header))
        self._check_success()

    def _check_success(self):
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_result == 0):
            #print(f"{self.name}: SUCCESS")
            self.send_signal(Signals.SUCCESS, self._qmem_pos[0])
            self.local_qcount = 0
            self.remote_meas_result = None
        elif self.local_meas_result == 0 and self.local_qcount > self.remote_qcount:
            pass
        else:
            self._handle_fail()
            #print(f"{self.name}: FAIL")
            self.send_signal(Signals.FAIL, self.local_qcount)
            self.local_meas_result = None
            self.remote_meas_result = None
    
    def _handle_fail(self):
        positions = [pos for pos in self._qmem_pos if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_pos = [None] * len(self._qmem_pos)

class QuantumDispatcher(NodeProtocol):

    def __init__(self, node, port_in, name="quantum_dispatcher"):
        super().__init__(node, name=name)
        self.port_in = port_in

    def run(self):
        while True:
            # Aliceから量子ビットが届くまで待つ
            yield self.await_port_input(self.port_in)
            msg = self.port_in.rx_input()

            if msg is None:
                print(f"{self.name}: Message is None")
                continue

            #print(f"{self.name}: Message received = {msg}")
            #print(f"{self.name}: Message metadata = {msg.meta}")

            pair_id = msg.meta.get("pair_id")
            if pair_id is None:
                #print(f"{self.name}: Message has no pair_id. Ignoring this message.")
                continue
            alice_mem_pos = msg.meta.get("mem_pos")
            protocol = msg.meta.get("protocol")

            #print(f"{self.name}: pair_id = {pair_id}")
            #print(f"{self.name}: Alice mem_pos = {alice_mem_pos}")
            #print(f"{self.name}: protocol = {protocol}")

            # ==================================================
            # pair_idに応じてBobのmemory positionを決定
            # ==================================================

            if pair_id == 1:
                bob_mem_pos = 0
                #print(f"{self.name}: Dispatching pair 1 -> Bob qmemory[{bob_mem_pos}]")

            elif pair_id == 2:
                bob_mem_pos = 1
                #print(f"{self.name}: Dispatching pair 2 -> Bob qmemory[{bob_mem_pos}]")

            else:
                #print(f"{self.name}: Unknown pair_id = {pair_id}")
                continue

            # ==================================================
            # Bobのqmemoryへ格納
            # ==================================================

            qubits = msg.items

            self.node.qmemory.put(qubits, positions=[bob_mem_pos])
            #print(f"{self.name}: Qubit stored at Bob qmemory[{bob_mem_pos}]")

            # ==================================================
            # RWMeasureへ到着通知
            # ==================================================

            self.send_signal(
                Signals.SUCCESS,
                {
                    "pair_id": pair_id,
                    "bob_mem_pos": bob_mem_pos,
                    "alice_mem_pos": alice_mem_pos,
                    "protocol": protocol
                }
            )

class Bennet(NodeProtocol):
    def __init__(self, node, port, role, start_expression=None, msg_header="bennet", name=None):   # 初期化
        if role.upper() not in ["A", "B"]:
            raise ValueError
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "BennetNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        self.role = role
        self._rotprog = self._rotate_program()
        self._measprog = self._measure_program()
        self._reprog = self._reverse_program()
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

    def _rotate_program(self):   # 回転操作（Aliceのみ）
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_Y, [q1])
        prog.apply(INSTR_Y, [q2])
        return prog
    
    def _measure_program(self):   # 測定（Alice & Bob）
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_CNOT, [q1, q2])
        prog.apply(INSTR_MEASURE, q2, output_key="M", inplace=False)
        return prog   # 測定結果をreturn

    def _reverse_program(self):   # 逆回転（Aliceのみ）
        prog = QuantumProgram(num_qubits=1)
        q1, = prog.get_qubit_indices(1)
        prog.apply(INSTR_Y, q1)
        return prog

    def run(self):
        #print(f"{self.name}:Start")
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression

        while True:
            expr = yield cchannel_ready | qmemory_ready
            print(f"{self.name}: first_term={expr.first_term.value}, second_term={expr.second_term.value}, time={sim_time()}")

            # ==================================================
            # Bobから古典測定結果を受信した場合
            # ==================================================
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
                    print(f"{self.name}: result {classical_message.items} received")

            # ==================================================
            # Bennett精製の開始条件が成立した場合
            # ==================================================
            elif expr.second_term.value:
                print(f"{self.name}: Bennett purification start / time: {sim_time()}")

                # ----------------------------------------------
                # 使用する量子ビットのメモリ位置を設定
                # ----------------------------------------------
                if self.name == "bennet_A":
                    self._qmem_positions = [0, 2]
                else:
                    self._qmem_positions = [0, 1]
                self.local_qcount += 1
                print(f"{self.name}: Bennett qubits = {self._qmem_positions}")

                # 2組のエンタングルメントを用いて
                # Bennett型エンタングルメント精製を実行
                yield from self._node_do_bennet()

            # ==================================================
            # 測定結果が揃ったか確認
            # ==================================================
            self._check_success()
            print(f"{self.name}: waiting again")
    
    def start(self):
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self):   # 失敗した場合、エンタングルメントを破棄
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]

    def _node_do_bennet(self):   # 精製処理
        self.num_runs += 1
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        #if self.role.upper() == "A":   # Aliceの場合、回転操作を行う
            #yield self.node.qmemory.execute_program(self._rotprog, [pos1, pos2])
        yield self.node.qmemory.execute_program(self._measprog, [pos1, pos2])   # 測定
        #if self.role.upper() == "A":   # Aliceの場合、回転操作を行う
            #yield self.node.qmemory.execute_program(self._reprog, [pos2])
        self.local_meas_result = self._measprog.output["M"][0]
        self._qmem_positions[1] = None
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))   # 測定結果をBobに送信
        
    def _check_success(self):   # 測定結果の比較
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                self.send_signal(Signals.SUCCESS, [self._qmem_positions[0], self.num_runs])
                print(f"{self.name}: SUCCESS / time: {sim_time()}")
                self.num_runs = 0
            else:
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
                print(f"{self.name}: FAIL")
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
    
class ProtectBennet(LocalProtocol):
    def __init__(self, node_a, node_b, num_runs, omega, theta):
            super().__init__(nodes={"A": node_a, "B": node_b}, name="Protect&Bennet")
            self.num_runs = num_runs
            # エンタングルメント生成プロトコル
            self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A1", input_mem_pos0=0,
                                               input_mem_pos1=1, num_pairs=1, name="entangle_A1"))
            self.add_subprotocol(LocalEntangle(node=node_a, qsource_name="QSource_A2", input_mem_pos0=2,
                                               input_mem_pos1=3, num_pairs=1, name="entangle_A2"))
            # 保護処理プロトコル
            self.add_subprotocol(Protect(node_a, node_a.ports["cout_bob1"], omega=omega, pair_id=1, name="protect_A1"))
            self.add_subprotocol(Protect(node_a, node_a.ports["cout_bob2"], omega=omega,pair_id=2, name="protect_A2"))
            self.add_subprotocol(QuantumDispatcher(node_b, node_b.ports["qdispatch_in"], name="quantum_dispatcher"))
            self.add_subprotocol(RWMeasure(node_b, node_b.ports["cin_alice1"], self.subprotocols["quantum_dispatcher"],
                                         pair_id=1, theta=theta, name="rwmeasure_B1"))
            self.add_subprotocol(RWMeasure(node_b, node_b.ports["cin_alice2"], self.subprotocols["quantum_dispatcher"],
                                         pair_id=2, theta=theta, name="rwmeasure_B2"))
            # 精製処理プロトコル
            self.add_subprotocol(Bennet(node_a, node_a.ports["cout_bob"], role="A", name="bennet_A"))
            self.add_subprotocol(Bennet(node_b, node_b.ports["cin_alice"], role="B", name="bennet_B"))
            # テレポーテーションプロトコル
            self.add_subprotocol(BellMeasurement(node=node_a, port=node_a.ports["cout_bob"], name="teleport_A"))
            self.add_subprotocol(Correction(node=node_b, name="teleport_B"))

            # エンタングルメント生成プロトコルの開始条件
            self.subprotocols["entangle_A1"].start_expression = (
                                    self.subprotocols["entangle_A1"].await_signal(self, Signals.WAITING) |
                                    self.subprotocols["entangle_A1"].await_signal(self.subprotocols["protect_A1"], Signals.FAIL) |
                                    self.subprotocols["entangle_A1"].await_signal(self.subprotocols["bennet_A"], Signals.FAIL))
            self.subprotocols["entangle_A2"].start_expression = (
                                    self.subprotocols["entangle_A2"].await_signal(self.subprotocols["protect_A1"], Signals.SUCCESS) |
                                    self.subprotocols["entangle_A2"].await_signal(self.subprotocols["protect_A2"], Signals.FAIL))
                                        
            # 保護処理プロトコルの開始条件                        
            self.subprotocols["protect_A1"].start_expression = (
                self.subprotocols["protect_A1"].await_signal(self.subprotocols["entangle_A1"], Signals.SUCCESS))
            self.subprotocols["protect_A2"].start_expression = (
                self.subprotocols["protect_A2"].await_signal(self.subprotocols["entangle_A2"], Signals.SUCCESS))
            self.subprotocols["rwmeasure_B1"].start_expression = (
                self.subprotocols["rwmeasure_B1"].await_signal(self, Signals.WAITING) |
                self.subprotocols["rwmeasure_B1"].await_signal(self.subprotocols["protect_A1"], Signals.FAIL))
            self.subprotocols["rwmeasure_B2"].start_expression = (
                            self.subprotocols["rwmeasure_B2"].await_signal(self.subprotocols["entangle_A1"], Signals.SUCCESS) |
                            self.subprotocols["rwmeasure_B2"].await_signal(self.subprotocols["protect_A2"], Signals.FAIL))

            # 精製処理プロトコルの開始条件                        
            self.subprotocols["bennet_A"].start_expression = (
                self.subprotocols["bennet_A"].await_signal(self.subprotocols["protect_A2"], Signals.SUCCESS))
                                                            
            self.subprotocols["bennet_B"].start_expression = (
                self.subprotocols["bennet_B"].await_signal(self.subprotocols["rwmeasure_B2"], Signals.SUCCESS))  

            # テレポーテーションプロトコルの開始条件
            self.subprotocols["teleport_A"].start_expression = self.subprotocols["teleport_A"].await_signal(
                                                                self.subprotocols["bennet_A"], Signals.SUCCESS)
            self.subprotocols["teleport_B"].start_expression = self.subprotocols["teleport_B"].await_signal(
                                                                self.subprotocols["bennet_B"], Signals.SUCCESS)
                                                                                                         

    def run(self):
            self.start_subprotocols()
            for i in range(self.num_runs):
                #print(f"Simulation {i}")
                start_time = sim_time()
                self.subprotocols["entangle_A1"].entangled_pairs = 0
                self.subprotocols["entangle_A2"].entangled_pairs = 0
                self.send_signal(Signals.WAITING)
                yield (self.await_signal(self.subprotocols["teleport_A"], Signals.SUCCESS)
                       & self.await_signal(self.subprotocols["teleport_B"], Signals.SUCCESS))
                signal_A = self.subprotocols["bennet_A"].get_signal_result(Signals.SUCCESS, self)
                result_en = {
                    "pairs1": self.subprotocols["entangle_A1"].entangled_pairs,
                    "pairs2": self.subprotocols["entangle_A2"].entangled_pairs,
                    "runs": signal_A[1],
                    "time": sim_time() - start_time
                }
                result_A = self.subprotocols["teleport_A"].get_signal_result(Signals.SUCCESS, self)
                result_B = self.subprotocols["teleport_B"].get_signal_result(Signals.SUCCESS, self)
                result_tel = {
                    "pos_A0": result_A["pos_A0"],
                    "pos_A1": result_A["pos_A1"],
                    "pos_B": result_B,
                    "time": sim_time() - start_time
                }
                self.send_signal(Signals.SUCCESS, [result_en, result_tel])

def sim_setup(node_a, node_b, num_runs, omega, theta):
    pb_example = ProtectBennet(node_a, node_b, num_runs, omega, theta)
    
    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result_en, result_tel = protocol.get_signal_result(Signals.SUCCESS)
        print(result_en)
        node_a.qmemory.discard(positions=[result_tel["pos_A0"]]) # 使用しているメモリを解放
        node_a.qmemory.discard(positions=[result_tel["pos_A1"]])
        q_B, = node_b.qmemory.pop(positions=[result_tel["pos_B"]])
        #print(qapi.reduced_dm([q_A, q_B]))
        pairs = result_en["pairs1"] + result_en["pairs2"]
        prob = 1 / result_en["runs"]
        f2 = qapi.fidelity(q_B, ks.y0, squared=True)
        return {"fidelity": f2, "pairs": pairs, "probability": prob, "time": result_en["time"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                        include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=pb_example,
                                        event_type=Signals.SUCCESS.value))
    return pb_example, dc

def network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=200, node_distance=200):
    network = Network("wmeasure_network")

    # ノード設定
    node_a, node_b = network.add_nodes(["node_A", "node_B"])
    node_a.add_subcomponent(QuantumProcessor("QuantumMemory_A", num_positions=6,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能
    state_sampler = StateSampler([ks.b00, ks.s00], probabilities=[source_fidelity_sq, 1 - source_fidelity_sq])
    source_frequency = 4e4 / node_distance
    node_a.add_subcomponent(QSource("QSource_A1", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_a.add_subcomponent(QSource("QSource_A2", state_sampler=state_sampler,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=2, status=SourceStatus.EXTERNAL))
    node_b.add_subcomponent(QuantumProcessor("QuantumMemory_B", num_positions=6,
        fallback_to_nonphysical=True))   # パラメータ「memory_noise_models」によりメモリ滞在によるノイズの影響を設定可能

    # チャネル設定
    conn_cchannel1 = DirectConnection("CChannelConn_AB_1",
        ClassicalChannel("CChannel_A1->B1", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B1->A1", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchannel1, label="channel1",
                           port_name_node1="cout_bob1", port_name_node2="cin_alice1")
    conn_cchanne2 = DirectConnection("CChannelConn_AB_2",
        ClassicalChannel("CChannel_A2->B2", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B2->A2", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchanne2, label="channel2",
                            port_name_node1="cout_bob2", port_name_node2="cin_alice2")
    conn_cchanne3 = DirectConnection("CChannelConn_AB_3",
            ClassicalChannel("CChannel_A3->B3", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}),
            ClassicalChannel("CChannel_B3->A3", length=node_distance, models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=conn_cchanne3, label="channel3",
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
    node_a.subcomponents["QSource_A1"].ports["qout0"].connect(
        node_a.qmemory.ports["qin0"])
    node_a.subcomponents["QSource_A1"].ports["qout1"].connect(
        node_a.qmemory.ports["qin1"])
    node_a.subcomponents["QSource_A2"].ports["qout0"].connect(
        node_a.qmemory.ports["qin2"])
    node_a.subcomponents["QSource_A2"].ports["qout1"].connect(
        node_a.qmemory.ports["qin3"])
    node_a.qmemory.ports["qout"].forward_output(node_a.ports["qout_bob"])
    # Link Bob ports:
    node_b.add_ports([
        "qdispatch_in",
    ])
    node_b.ports["qin_alice"].forward_input(
        node_b.ports["qdispatch_in"]
    )
    return network

if __name__ == "__main__":
    network = network_setup()
    pb_example, dc = sim_setup(network.get_node("node_A"), network.get_node("node_B"), 3, np.pi/3, 0.2)
    pb_example.start()
    ns.sim_run()
    print("Fidelity of generated entanglement: {}".format(dc.dataframe["fidelity"].mean()))