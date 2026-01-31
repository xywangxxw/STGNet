import torch
import torch.nn as nn
from torch_geometric.utils import degree

from utils import (
    transform_point_to_local_coordinate,
    compute_angles_lengths_2D,
    wrap_angle,
)


class DynamicNeighboringModule(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_modes: int,
                 radius: float,
                 embedding_layer: nn.Module,
                 graph_attention_layer: nn.Module,
                 in_degree_project: nn.Module,
                 length_project: nn.Module,
                 radius_offset: nn.Module,
                 drop_edge_strategy: nn.Module):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.radius = radius

        self.embedding_layer = embedding_layer
        self.margin_graph_attention = graph_attention_layer
        self.in_degree_project = in_degree_project
        self.length_project = length_project
        self.radius_offset = radius_offset
        self.drop_edge_strategy = drop_edge_strategy

    def _select_margin_edges(self, edge_index, positions):
        src, tgt = edge_index
        dist = torch.norm(positions[tgt] - positions[src], p=2, dim=-1)
        mask = (dist <= self.radius) & (dist >= 0.5 * self.radius)
        return edge_index[:, mask], dist[mask]

    def _margin_interaction(self, node_embs, edge_index, positions, headings, num_all_agent):
        edge_vec = transform_point_to_local_coordinate(
            point=positions[edge_index[0]],
            position=positions[edge_index[1]],
            heading=headings[edge_index[1]],
        )
        edge_len, edge_theta = compute_angles_lengths_2D(edge_vec)
        edge_heading = wrap_angle(headings[edge_index[0]] - headings[edge_index[1]])

        edge_attr = torch.stack([
            edge_len,
            edge_theta.sin(),
            edge_theta.cos(),
            edge_heading.sin(),
            edge_heading.cos()
        ], dim=-1)

        edge_attr_embs = self.embedding_layer(edge_attr)

        node_embs_reshape = node_embs.reshape(num_all_agent, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1, self.hidden_dim)

        margin_node_embs = self.margin_graph_attention(
            x=node_embs_reshape,
            edge_index=edge_index,
            edge_attr=edge_attr_embs
        )

        return margin_node_embs

    def _dynamic_radius_refine(self,
                               margin_node_embs,
                               edge_index,
                               edge_attr,
                               rel_distance,
                               node_lengths,
                               num_nodes):

        tgt_nodes = edge_index[1]
        tgt_in_degree = degree(tgt_nodes, num_nodes=num_nodes).unsqueeze(-1)
        tgt_length_feat = node_lengths.unsqueeze(-1).permute(1,0,2).reshape(-1,1)

        deg_feat = self.in_degree_project(tgt_in_degree)
        len_feat = self.length_project(tgt_length_feat)

        radius_offset = self.radius_offset(torch.cat([margin_node_embs, deg_feat, len_feat], dim=-1))

        new_edge_index, new_edge_attr = self.drop_edge_strategy(
            edge_index=edge_index,
            edge_attr=edge_attr,
            rel_distance=rel_distance,
            offset=radius_offset
        )
        return new_edge_index, new_edge_attr

    def forward(self, node_embs, edge_index, edge_attr, positions, headings, node_lengths, num_all_agent, num_nodes):
        margin_edge_index, margin_edge_len = self._select_margin_edges(edge_index, positions)

        margin_node_embs = self._margin_interaction(
            node_embs, margin_edge_index, positions, headings, num_all_agent
        )

        rel_distance_full = torch.norm(positions[edge_index[1]] - positions[edge_index[0]], p=2, dim=-1)

        num_nodes = self.num_modes * num_nodes
        new_edge_index, new_edge_attr = self._dynamic_radius_refine(
            margin_node_embs,
            edge_index,
            edge_attr,
            rel_distance_full,
            node_lengths,
            num_nodes
        )

        return new_edge_index, new_edge_attr