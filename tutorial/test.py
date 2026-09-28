import netsquid as ns

from netsquid.nodes import Node
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel
from netsquid.components.component import Message
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.protocols.nodeprotocols import NodeProtocol


# ============================================================
# 古典メッセージ受信プロトコル
# ============================================================
class ClassicalReceiver(NodeProtocol):

    def __init__(self, node, port, name):
        super().__init__(
            node=node,
            name=name
        )

        self.port = port

    def run(self):

        print(
            f"{self.name}: START "
            f"(waiting on {self.port.name})"
        )

        while True:

            # 古典チャネルからの入力を待つ
            yield self.await_port_input(self.port)

            print(
                f"{self.name}: "
                f"CLASSICAL MESSAGE EVENT "
                f"on {self.port.name}"
            )

            # メッセージを取得
            msg = self.port.rx_input()

            print(
                f"{self.name}: "
                f"Received = {msg}"
            )


# ============================================================
# 古典メッセージ送信プロトコル
# ============================================================
class SenderProtocol(NodeProtocol):

    def __init__(
        self,
        node,
        port,
        message,
        header,
        delay=0,
        name=None
    ):
        super().__init__(node=node, name=name)

        self.port = port
        self.message = message
        self.header = header
        self.delay = delay

    def run(self):

        print(
            f"{self.name}: START at "
            f"{ns.sim_time()}"
        )

        if self.delay > 0:
            yield self.await_timer(
                self.delay
            )

        print(
            f"{self.name}: Sending "
            f"{self.message} at "
            f"{ns.sim_time()}"
        )

        self.port.tx_output(
            ns.components.Message(
                items=self.message,
                header=self.header
            )
        )

        print(
            f"{self.name}: Finished sending at "
            f"{ns.sim_time()}"
        )


# ============================================================
# ネットワーク構築
# ============================================================
def network_setup(node_distance=10):

    network = Network(
        "classical_test_network"
    )

    # --------------------------------------------------------
    # ノード作成
    # --------------------------------------------------------
    node_a, node_b = network.add_nodes(
        [
            "node_A",
            "node_B"
        ]
    )

    # --------------------------------------------------------
    # 古典チャネル1
    #
    # Alice
    #   cout_bob1
    #       ↓
    #   CChannel_A1_B1
    #       ↓
    #   cin_alice1
    #       ↓
    # Bob
    # --------------------------------------------------------
    conn_cchannel1 = DirectConnection(
        "CChannelConn_Test1",

        ClassicalChannel(
            "CChannel_A1_B1",
            length=node_distance,
            models={
                "delay_model": FibreDelayModel(
                    c=200e3
                )
            }
        ),

        ClassicalChannel(
            "CChannel_B1_A1",
            length=node_distance,
            models={
                "delay_model": FibreDelayModel(
                    c=200e3
                )
            }
        )
    )


    # --------------------------------------------------------
    # 古典チャネル2
    #
    # Alice
    #   cout_bob2
    #       ↓
    #   CChannel_A2_B2
    #       ↓
    #   cin_alice2
    #       ↓
    # Bob
    # --------------------------------------------------------
    conn_cchannel2 = DirectConnection(
        "CChannelConn_Test2",

        ClassicalChannel(
            "CChannel_A2_B2",
            length=node_distance,
            models={
                "delay_model": FibreDelayModel(
                    c=200e3
                )
            }
        ),

        ClassicalChannel(
            "CChannel_B2_A2",
            length=node_distance,
            models={
                "delay_model": FibreDelayModel(
                    c=200e3
                )
            }
        )
    )

    node_a.add_ports([
        "cout_bob1",
        "cout_bob2"
    ])

    node_b.add_ports([
        "cin_alice1",
        "cin_alice2"
    ])

    network.add_connection(
        node_a,
        node_b,
        connection=conn_cchannel1,
        label="channel1",
        port_name_node1="cout_bob2",
        port_name_node2="cin_alice2"
    )

    network.add_connection(
        node_a,
        node_b,
        connection=conn_cchannel2,
        label="channel2",
        port_name_node1="cout_bob1",
        port_name_node2="cin_alice1"
    )
    return network


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # シミュレーションをリセット
    # --------------------------------------------------------
    ns.sim_reset()

    # --------------------------------------------------------
    # ネットワーク構築
    # --------------------------------------------------------
    network = network_setup(
        node_distance=10
    )

    node_a = network.get_node(
        "node_A"
    )

    node_b = network.get_node(
        "node_B"
    )

    # --------------------------------------------------------
    # ポート取得
    # --------------------------------------------------------
    alice_port1 = node_a.ports[
        "cout_bob1"
    ]

    bob_port1 = node_b.ports[
        "cin_alice1"
    ]

    alice_port2 = node_a.ports[
        "cout_bob2"
    ]

    bob_port2 = node_b.ports[
        "cin_alice2"
    ]

    # --------------------------------------------------------
    # ポート確認
    # --------------------------------------------------------
    print()
    print("========================================")
    print("           PORT CHECK")
    print("========================================")

    print(
        f"Channel 1:"
    )

    print(
        f"  Alice port = {alice_port1}"
    )

    print(
        f"  Bob port   = {bob_port1}"
    )

    print()

    print(
        f"Channel 2:"
    )

    print(
        f"  Alice port = {alice_port2}"
    )

    print(
        f"  Bob port   = {bob_port2}"
    )

    print(
        "========================================"
    )
    print()

    # --------------------------------------------------------
    # Receiverを2つ作成
    # --------------------------------------------------------

    receiver_B1 = ClassicalReceiver(
        node=node_b,
        port=bob_port1,
        name="receiver_B1"
    )

    receiver_B2 = ClassicalReceiver(
        node=node_b,
        port=bob_port2,
        name="receiver_B2"
    )

    # --------------------------------------------------------
    # Senderを2つ作成
    # --------------------------------------------------------

    sender_A1 = SenderProtocol(
        node=node_a,
        port=alice_port1,
        message=[111],
        header="test1",
        delay=0,
        name="sender_A1"
    )

    sender_A2 = SenderProtocol(
        node=node_a,
        port=alice_port2,
        message=[222],
        header="test2",
        delay=0,
        name="sender_A2"
    )

    # --------------------------------------------------------
    # Protocol開始
    # --------------------------------------------------------

    receiver_B1.start()
    receiver_B2.start()

    sender_A1.start()
    sender_A2.start()

    # --------------------------------------------------------
    # シミュレーション実行
    # --------------------------------------------------------

    print(
        "===== SIMULATION START ====="
    )

    ns.sim_run()

    print(
        "===== SIMULATION END ====="
    )