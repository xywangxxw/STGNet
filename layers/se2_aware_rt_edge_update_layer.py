import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from utils import wrap_angle
from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate


class SE2AwareEdgeUpdateLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 edge_dim: int = 5):
        super().__init__()

        self.geo_embed = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.geo_pair_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        self.semantic_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.edge_norm = nn.LayerNorm(hidden_dim)

        self.edge_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.edge_residual = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def get_raw_edge(self,
                     src_node_idx: torch.Tensor,
                     tgt_node_idx: torch.Tensor,
                     positions_i: torch.Tensor,
                     positions_j: torch.Tensor,
                     headings_i: torch.Tensor,
                     headings_j: torch.Tensor):

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
                positions_i: torch.Tensor,
                positions_j: torch.Tensor,
                headings_i: torch.Tensor,
                headings_j: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                node_embs_i: torch.Tensor,
                node_embs_j: torch.Tensor,
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

        g_ij = self.geo_embed(raw_e_ij)
        g_ji = self.geo_embed(raw_e_ji)

        geo_msg = F.relu(self.geo_pair_proj(torch.cat([g_ij, g_ji], dim=-1)))

        n_i = node_embs_i[edge_index[0]]
        n_j = node_embs_j[edge_index[1]]

        gate = torch.sigmoid(self.semantic_gate(torch.cat([n_i, n_j], dim=-1)))
        geo_msg = geo_msg * gate

        old_e = edge_attr
        h = self.edge_norm(geo_msg + old_e)

        update_gate = torch.sigmoid(self.edge_gate(h))
        new_e = h + update_gate * (self.edge_residual(old_e) - h)

        return new_e
