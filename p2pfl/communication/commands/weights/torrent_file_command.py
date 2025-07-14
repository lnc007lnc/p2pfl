import os
import time
import libtorrent as lt

from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.node_state import NodeState
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.management.logger import logger
import threading
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.stages.base_node.gossip_model_stage import GossipModelStage

from tqdm import tqdm


class TorrentFileCommand(FullModelCommand):
    """通过 Gossip 交换 .torrent 元数据，并通过 BitTorrent 下载模型权重"""

    @staticmethod
    def get_name() -> str:
        return "TorrentFileCommand"

    def __init__(
        self,
        state: NodeState,
        stop_fn,
        aggregator: Aggregator,
        learner: Learner,
        communication_protocol: CommunicationProtocol,
    ) -> None:
        super().__init__(state, stop_fn, aggregator, learner)
        self.communication_protocol = communication_protocol

    def execute(self, *args, **kwargs) -> None:
        # —— 1) 从 args/kwargs 中解析 torrent_bytes, rnd, sender ——
        logger.info(self.state.addr, "Received torrent command.")

        torrent_bytes = None
        # 优先从 kwargs
        if "weights" in kwargs:
            torrent_bytes = kwargs["weights"]
        # 然后尝试在 args 里找 bytes
        if torrent_bytes is None:
            for a in args:
                if isinstance(a, (bytes, bytearray)):
                    torrent_bytes = a
                    break
        if torrent_bytes is None:
            raise ValueError("TorrentFileCommand.execute: missing torrent data")

        # 解析轮次
        rnd = kwargs.get("round", None)
        if rnd is None:
            for a in args:
                if isinstance(a, int):
                    rnd = a
                    break
        if rnd is None:
            raise ValueError("TorrentFileCommand.execute: missing round number")

        # 解析发送者
        sender = kwargs.get("sender", None)
        if sender is None:
            for a in args:
                if isinstance(a, str):
                    sender = a
                    break
        if sender is None:
            sender = ""  # 或者使用 self.state.addr 作为默认

        # —— 2) de-dup by info_hash ——
        info = lt.torrent_info(lt.bdecode(torrent_bytes))
        info_hash = info.info_hash()
        if info_hash in self.state.received_torrents:
            return
        self.state.received_torrents.add(info_hash)

        # —— 3) save .torrent ——
        torrent_dir = os.path.join(os.getcwd(), "temp_torrents")
        os.makedirs(torrent_dir, exist_ok=True)
        #torrent_path = os.path.join(torrent_dir, f"{self.state.exp_name}_{rnd}.torrent")
        torrent_path = os.path.join(torrent_dir, f"round{rnd}_{info_hash}.torrent")
        with open(torrent_path, "wb") as f:
            f.write(torrent_bytes)
        logger.info(self.state.addr, f"[Torrent] Saved torrent to {torrent_path}")

        # 3) 首次接收本轮种子时，生成并广播自己的 torrent
        if rnd not in self.state.generated_torrent_rounds:

            self.state.generated_torrent_rounds.add(rnd)
            # GossipModelStage._GossipModelStage__gossip_model_difusion(
            #     self.state,
            #     self.communication_protocol,
            #     self.learner,
            # )
            GossipModelStage.execute(state=self.state, communication_protocol=self.communication_protocol,aggregator=self.aggregator, learner=self.learner)

        # 4) 异步启动下载线程
        t = threading.Thread(
            target=self._download_and_store,
            args=(torrent_path, rnd, sender),
            daemon=True
        )
        t.start()



        # 4) 立即返回，不阻塞网络线程
        return


    def _download_and_store(self, torrent_path: str, rnd: int, sender: str):
        """后台下载并调用父类添加模型的方法。"""
        # —— 4) download via BitTorrent ——
        ses = lt.session()

        save_dir = os.path.join(os.getcwd(), "temp_models")
        os.makedirs(save_dir, exist_ok=True)

        ti = lt.torrent_info(torrent_path)
        torrent_params = {
            "ti": ti,
            "save_path": save_dir,
        }
        handle = ses.add_torrent(torrent_params)

        # 等待 metadata 完全获取
        while not handle.has_metadata():
            time.sleep(1)
        # 等待所有 piece 下载完成
        s = handle.status()
        pbar = tqdm(total=1.0)
        last_progress = 0.0
        while handle.status().state != lt.torrent_status.seeding:
            s = handle.status()
            progress = s.progress
            pbar.update(progress - last_progress)
            last_progress = progress
            time.sleep(1)
        pbar.close()
        logger.info(self.state.addr, f"[Torrent] Download complete for round {rnd}")

        # —— 5) 读取下载的模型文件并交给聚合逻辑 ——
        files = ti.files()
        rel_path = files.file_path(0)
        model_file = os.path.join(save_dir, rel_path)
        with open(model_file, "rb") as mf:
            model_bytes = mf.read()
        super().execute(
            source=sender,
            round=rnd,
            weights=model_bytes
        )
        self.state.download_event.set()