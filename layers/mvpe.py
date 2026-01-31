import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from layers import MLP
from utils import wrap_angle
from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate


class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=True)

    def forward(self, x):
        return self.up(self.down(x))


class MutualViewpointPositionalEncoding(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 edge_dim: int = 5,
                 msg_rank_ratio: float = 0.5,
                 ffn_rank_ratio: float = 0.5):
        super().__init__()

        msg_rank = int(hidden_dim * msg_rank_ratio)
        ffn_rank = int(hidden_dim * ffn_rank_ratio)

        self.raw_edge_embedding = MLP(
            input_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )

        self.m_linear_embedding = LowRankLinear(
            in_dim=hidden_dim * 4,
            out_dim=hidden_dim,
            rank=msg_rank
        )

        self.u_norm = nn.LayerNorm(hidden_dim)
        self.u_linear_embedding = nn.Linear(hidden_dim, hidden_dim)

        self.e_norm = nn.LayerNorm(hidden_dim)

        self.e_ffn_down = nn.Linear(hidden_dim, ffn_rank)
        self.e_ffn_up = nn.Linear(ffn_rank, hidden_dim)

        self.old_edge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.new_edge_proj = nn.Linear(hidden_dim, hidden_dim)

    def get_raw_edge(self,
                     src_node_idx,
                     tgt_node_idx,
                     positions_i,
                     positions_j,
                     headings_i,
                     headings_j):

        raw_vec = transform_point_to_local_coordinate(
            point=positions_i[src_node_idx],
            position=positions_j[tgt_node_idx],
            heading=headings_j[tgt_node_idx]
        )
        length, theta = compute_angles_lengths_2D(raw_vec)
        heading = wrap_angle(headings_i[src_node_idx] - headings_j[tgt_node_idx])

        return torch.stack([
            length,
            theta.cos(),
            theta.sin(),
            heading.cos(),
            heading.sin()
        ], dim=-1)

    def forward(self,
                positions_i,
                positions_j,
                headings_i,
                headings_j,
                edge_index,
                edge_attr,
                node_embs_i,
                node_embs_j,
                time_intervals: Optional[torch.Tensor] = None):

        raw_e_ij = self.get_raw_edge(
            edge_index[0], edge_index[1],
            positions_i, positions_j,
            headings_i, headings_j
        )
        raw_e_ji = self.get_raw_edge(
            edge_index[1], edge_index[0],
            positions_j, positions_i,
            headings_j, headings_i
        )

        if time_intervals is not None:
            raw_e_ij = torch.cat([raw_e_ij, time_intervals.unsqueeze(-1)], dim=-1)
            raw_e_ji = torch.cat([raw_e_ji, -time_intervals.unsqueeze(-1)], dim=-1)

        e_ij = self.raw_edge_embedding(raw_e_ij)
        e_ji = self.raw_edge_embedding(raw_e_ji)

        n_i = node_embs_i[edge_index[0]]
        n_j = node_embs_j[edge_index[1]]

        m_ij = F.relu(self.m_linear_embedding(
            torch.cat([e_ij, e_ji, n_i, n_j], dim=-1)
        ))

        u_ij = self.u_norm(self.u_linear_embedding(m_ij) + e_ij)

        ffn = self.e_ffn_up(F.relu(self.e_ffn_down(u_ij)))
        new_e_ij = self.e_norm(ffn + u_ij)

        old_proj = self.old_edge_proj(edge_attr)
        gate = torch.sigmoid(old_proj + self.new_edge_proj(new_e_ij))

        return new_e_ij + gate * (old_proj - new_e_ij)
