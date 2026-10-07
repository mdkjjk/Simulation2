import netsquid as ns

# 必要となるパッケージのインポート
import netsquid.components.instructions as instr
from netsquid.components.component import Message
from netsquid.components.qprogram import QuantumProgram
from netsquid.util.simtools import sim_time
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from pydynaa import EventExpression


class InitStateProgram(QuantumProgram): # 伝送する量子ビットを生成するプログラム
    default_num_qubits = 1

    def program(self):
        q1, = self.get_qubit_indices(1)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_S, q1)
        yield self.run()


class BellMeasurement(NodeProtocol): # 量子テレポーテーションにおけるAlice側での処理を行うプロトコル
    def __init__(self, node, port, name=None):
        super().__init__(node, name)
        self.port = port
        self._qmem_pos0 = None
        self._qmem_pos1 = None
        self.header = "teleport"

    def start(self):
        super().start()
        if self.start_expression is not None and not isinstance(self.start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(self.start_expression)))

    def run(self):
        qubit_initialised = False
        entanglement_ready = False
        qubit_init_program = InitStateProgram()
        while True:
            expr_port = self.start_expression
            yield expr_port # エンタングルメントの到着を待機
            entanglement_ready = True
            source_protocol = expr_port.atomic_source
            ready_signal = source_protocol.get_signal_by_event(event=expr_port.triggered_events[0], receiver=self)
            #print(ready_signal.result)
            if isinstance(ready_signal.result, list):
                self._qmem_pos1 = ready_signal.result[0] # エンタングルメントが保存された量子メモリポジションを取得
            else:
                self._qmem_pos1 = ready_signal.result
            #print(f"{self.name}: Entanglement received at {self._qmem_pos1} / time: {sim_time()}")

            # 空いている量子メモリポジションで伝送する量子ビットを生成
            self._qmem_pos0 = self.node.qmemory.unused_positions[0]
            self.node.qmemory.execute_program(qubit_init_program, qubit_mapping=[self._qmem_pos0])
            expr_signal = self.await_program(self.node.qmemory)
            yield expr_signal # 伝送する量子ビットの生成が完了するまで待機
            qubit_initialised = True
            #print(f"{self.name}: Initqubit received at {self._qmem_pos0} / time: {sim_time()}")

            # 上記の2つが完了した場合
            if qubit_initialised and entanglement_ready: 
                self.node.qmemory.operate(ns.CNOT, [self._qmem_pos0, self._qmem_pos1]) # CNOTゲート適用
                self.node.qmemory.operate(ns.H, self._qmem_pos0) # Hゲート適用
                m, _ = self.node.qmemory.measure([self._qmem_pos0, self._qmem_pos1]) # 測定
                self.port.tx_output(Message(m, header=self.header)) # 測定結果をBobへ送信
                result = {"pos_A0": self._qmem_pos0,
                          "pos_A1": self._qmem_pos1,}
                self.send_signal(Signals.SUCCESS, result) # プロトコルの成功シグナルと使用しているメモリポジションを送信
                #print(f"{self.name}: Finish / time: {sim_time()}")
                qubit_initialised = False
                entanglement_ready = False


class Correction(NodeProtocol): # 量子テレポーテーションにおけるBob側での処理を行うプロトコル
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
            expr = yield (self.await_port_input(port_alice) | expr_signal) # エンタングルメントの到着 or Aliceの測定結果の到着を待機

            # Aliceの測定結果が到着した場合
            if expr.first_term.value:
                msg = port_alice.rx_input(header="teleport") # 測定結果を取得
                if msg is not None:
                    meas_results = msg.items
                    #print(f"{self.name}: Result: {meas_results} / time: {sim_time()}")
            # エンタングルメントが到着した場合
            else:
                entanglement_ready = True
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(event=expr.second_term.triggered_events[-1], receiver=self)
                if isinstance(ready_signal.result, list):
                    self._qmem_pos = ready_signal.result[0] # エンタングルメントが保存された量子メモリポジションを取得
                else:
                    self._qmem_pos = ready_signal.result
                #print(f"{self.name}: Entanglement received at {self._qmem_pos} / time: {sim_time()}")

            # 上記の2つが完了した場合
            if meas_results is not None and entanglement_ready:
                # 測定結果に応じて、Z,Xゲートを適用
                if meas_results[0] == 0:
                    self.node.qmemory.execute_instruction(instr.INSTR_Z, [self._qmem_pos])
                if meas_results[1] == 0:
                    self.node.qmemory.execute_instruction(instr.INSTR_X, [self._qmem_pos])
                self.send_signal(Signals.SUCCESS, self._qmem_pos) # プロトコルの成功シグナルと使用しているメモリポジションを送信
                #print(f"{self.name}: Teleport success / time: {sim_time()}")
                entanglement_ready = False
                meas_results = None