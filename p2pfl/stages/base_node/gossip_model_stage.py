#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Gossip model stage."""

from typing import Any, List, Optional, Type, Union

from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory

import os
import libtorrent as lt
#from p2pfl.communication.commands.weights.torrent_file_command import TorrentFileCommand
import hashlib

class GossipModelStage(Stage):
    """Gossip model stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "GossipModelStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None,
        communication_protocol: Optional[CommunicationProtocol] = None,
        aggregator: Optional[Aggregator] = None,
        learner: Optional[Learner] = None,
        **kwargs,
    ) -> Union[Type["Stage"], None]:
        """Execute the stage."""
        if state is None or aggregator is None or communication_protocol is None or learner is None:
            raise Exception("Invalid parameters on GossipModelStage.")

        logger.info(state.addr, "🗣️ Gossiping aggregated model.")

        # 确保上一轮事件已被清除
        # —— 新增：关闭所有 seeding sessions ——
        for ses in getattr(state, "seed_sessions", []):
            # 把该 session 里的所有 torrent handle 移除，不删除已下载文件
            for th in ses.get_torrents():
                try:
                    ses.remove_torrent(th)
                except Exception as e:
                    logger.warning(state.addr, f"[Torrent] Failed to remove torrent: {e}")
            # 可选：暂停 session
            try:
                ses.pause()
            except Exception:
                pass
        # 清空列表
        state.seed_sessions = []

        GossipModelStage.__gossip_model_difusion(state, communication_protocol, learner)

        # 阻塞直到后台下载线程通过 FullModelCommand.execute 设置 event
        logger.info(state.addr, "⏳ Waiting for model download to complete...")
        state.download_event.wait()
        logger.info(state.addr, "✅ Model download and set_model finished.")

        return StageFactory.get_stage("RoundFinishedStage")

    @staticmethod
    def __gossip_model_difusion(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
    ) -> None:
        logger.info(state.addr, "🗣️ Gossiping aggregated model.")
        fixed_round = state.round
        if fixed_round is None:
            raise Exception("Learner not initialized")

        def candidate_condition(node: str) -> bool:
            return state.nei_status[node] < fixed_round

        def get_candidates_fn() -> List[str]:
            return [n for n in communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]

        def status_fn() -> Any:
            return get_candidates_fn()

        # def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
        #     if state.round is None:
        #         raise Exception("Round not initialized")
        #     encoded_model = learner.get_model().encode_parameters()
        #     return (
        #         communication_protocol.build_weights(FullModelCommand.get_name(), state.round, encoded_model),
        #         FullModelCommand.get_name(),
        #         state.round,
        #         [str(state.round)],
        #     )
        def model_fn(node: str) -> tuple[bytes, str, int, list[str]]:
            # 1. 取得编码后的模型权重
            encoded = learner.get_model().encode_parameters()

            data_hash = hashlib.sha1(encoded).hexdigest()

            # 2. 保存模型文件供 BitTorrent 使用
            src_dir = os.path.join(os.getcwd(), "bittorrent_source")
            os.makedirs(src_dir, exist_ok=True)
            model_filename = os.path.join(src_dir, f"round{state.round}_{data_hash}.pt")
            with open(model_filename, "wb") as mf:
                mf.write(encoded)

            # 3. 构建 torrent 元数据
            fs = lt.file_storage()
            lt.add_files(fs, model_filename)
            tor = lt.create_torrent(fs)
            tor.add_tracker("udp://tracker.openbittorrent.com:80")
            tor.add_tracker("udp://tracker.leechers-paradise.org:6969/announce")
            tor.add_tracker("http://tracker.opentrackr.org:1337/announce")
            tor.add_tracker("udp://tracker.coppersurfer.tk:6969/announce")
            # 关键：计算 piece hashes
            lt.set_piece_hashes(tor, src_dir)

            meta = tor.generate()
            torrent_data = lt.bencode(meta)

            # 4. 保存 .torrent 文件到本地，供 seeding & 调试
            info = lt.torrent_info(lt.bdecode(torrent_data))
            info_hash = info.info_hash()
            torrent_dir = os.path.join(os.getcwd(), "torrents")
            os.makedirs(torrent_dir, exist_ok=True)
            torrent_path = os.path.join(torrent_dir, f"round{state.round}_{info_hash}.torrent")
            with open(torrent_path, "wb") as f:
                f.write(torrent_data)
            logger.info(state.addr, f"[Torrent] Generated and saved torrent to {torrent_path}")

            # 5. 启动 seeding 会话，让本节点也做种
            ses = lt.session()
            #ses.listen_on(6881, 6891)
            ti = lt.torrent_info(torrent_path)
            ses.add_torrent({"ti": ti, "save_path": src_dir})
            # 保持 session 引用，避免被 GC 关闭
            state.seed_sessions.append(ses)

            # 6. 返回 payload，仍然通过 Gossip 交换 torrent_data

            payload = communication_protocol.build_weights(
                "TorrentFileCommand",
                state.round,
                torrent_data,
            )
            return (
                payload,
                "TorrentFileCommand",
                state.round,
                [str(state.round)],
            )

        # Gossip
        communication_protocol.gossip_weights(
            lambda: check_early_stop(state, raise_exception=False),
            get_candidates_fn,
            status_fn,
            model_fn,
        )
