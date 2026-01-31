from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import GraphAttention
from layers import MLP
from utils import compute_angles_lengths_2D
from utils import drop_edge_between_samples
from utils import init_weights
from utils import transform_point_to_local_coordinate
from utils import wrap_angle


class SceneEncoder(nn.Module):
    
    def __init__(self,
                 num_historical_steps: int,
                 pos_duration: int,
                 l2a_radius: float,
                 a2a_radius: float,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float):
        super(SceneEncoder, self).__init__()

        self.num_historical_steps = num_historical_steps
        self.pos_duration = pos_duration
        self.l2a_radius = l2a_radius
        self.a2a_radius = a2a_radius
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.a_embedding = MLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2a_edge_attr_embedding = MLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2a_edge_attr_embedding = MLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.a2a_edge_attr_embedding = MLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.t2a_graph_attention = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True)
        self.l2a_graph_attention = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.a2a_graph_attention = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_layers)])

        self.apply(init_weights)

    def forward(self,
                data: HeteroData,
                lane_embed: Dict):
        a_length = data['agent']['length']
        a_embs = self.a_embedding(a_length.unsqueeze(-1))
        num_all_agent = a_length.shape[0]

        a_batch = data['agent']['batch']
        a_position = data['agent']['position'][:, :self.num_historical_steps]
        a_heading = data['agent']['heading']
        a_valid_mask = data['agent']['valid_mask'][:, :self.num_historical_steps]

        t2a_position_t = data['agent']['position'][:,:self.num_historical_steps].reshape(-1, 2)
        t2a_position_a = a_position.reshape(-1, 2)
        t2a_heading_t = data['agent']['heading'].reshape(-1)
        t2a_heading_a  = a_heading.reshape(-1)
        t2a_valid_mask_t = data['agent']['valid_mask'][:, :self.num_historical_steps]
        t2a_valid_mask_a = a_valid_mask
        t2a_valid_mask = t2a_valid_mask_t.unsqueeze(2) & t2a_valid_mask_a.unsqueeze(1)
        t2a_edge_index = dense_to_sparse(t2a_valid_mask.contiguous())[0]
        t2a_edge_index = t2a_edge_index[:, t2a_edge_index[1] > t2a_edge_index[0]]
        t2a_edge_index = t2a_edge_index[:, t2a_edge_index[1] - t2a_edge_index[0] <= self.pos_duration]
        t2a_edge_vector = transform_point_to_local_coordinate(point=t2a_position_t[t2a_edge_index[0]],
                                                              position=t2a_position_a[t2a_edge_index[1]],
                                                              heading=t2a_heading_a[t2a_edge_index[1]])
        t2a_edge_length, t2a_edge_theta = compute_angles_lengths_2D(t2a_edge_vector)
        t2a_edge_heading = wrap_angle(t2a_heading_t[t2a_edge_index[0]] - t2a_heading_a[t2a_edge_index[1]])
        t2a_time_interval = t2a_edge_index[0] - t2a_edge_index[1]
        t2a_edge_attr_input = torch.stack([t2a_edge_length, t2a_edge_theta, t2a_edge_heading, t2a_time_interval], dim=-1)
        t2a_edge_attr_embs = self.t2a_edge_attr_embedding(t2a_edge_attr_input)

        l_embs = lane_embed['l_embs']
        l2a_position_l = data['lane']['position']
        l2a_position_a = a_position.reshape(-1, 2)
        l2a_heading_l = data['lane']['heading']
        l2a_heading_a = a_heading.reshape(-1)
        l2a_batch_l = data['lane']['batch']
        l2a_batch_a = a_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps,1).reshape(-1)
        l2a_valid_mask_l = data['lane']['visible_mask']
        l2a_valid_mask_a = a_valid_mask.reshape(-1)
        l2a_valid_mask = l2a_valid_mask_l.unsqueeze(1) & l2a_valid_mask_a.unsqueeze(0)
        l2a_valid_mask = drop_edge_between_samples(l2a_valid_mask, batch=(l2a_batch_l, l2a_batch_a))

        l2a_edge_index = dense_to_sparse(l2a_valid_mask.contiguous())[0]
        l2a_edge_index = l2a_edge_index[:, torch.norm(l2a_position_l[l2a_edge_index[0]] - l2a_position_a[l2a_edge_index[1]], p=2,
                                     dim=-1) < self.l2a_radius]
        l2a_edge_vector = transform_point_to_local_coordinate(point=l2a_position_l[l2a_edge_index[0]],
                                                              position=l2a_position_a[l2a_edge_index[1]],
                                                              heading=l2a_heading_a[l2a_edge_index[1]])
        l2a_edge_length, l2a_edge_theta = compute_angles_lengths_2D(l2a_edge_vector)
        l2a_edge_heading = wrap_angle(l2a_heading_l[l2a_edge_index[0]] - l2a_heading_a[l2a_edge_index[1]])
        l2a_edge_attr_input = torch.stack([l2a_edge_length, l2a_edge_theta, l2a_edge_heading], dim=-1)
        l2a_edge_attr_embs = self.l2a_edge_attr_embedding(l2a_edge_attr_input)

        a2a_position_a = a_position.permute(1, 0, 2).reshape(-1, 2)
        a2a_heading_a = a_heading.permute(1, 0).reshape(-1)
        a2a_batch_a = a_batch
        a2a_valid_mask = a_valid_mask.permute(1, 0)
        a2a_valid_mask = a2a_valid_mask.unsqueeze(2) & a2a_valid_mask.unsqueeze(1)
        a2a_valid_mask = drop_edge_between_samples(a2a_valid_mask, a2a_batch_a)

        a2a_edge_index = dense_to_sparse(a2a_valid_mask.contiguous())[0]
        a2a_edge_index = a2a_edge_index[:, a2a_edge_index[1] != a2a_edge_index[0]]
        a2a_edge_index = a2a_edge_index[:, torch.norm(a2a_position_a[a2a_edge_index[1]] - a2a_position_a[a2a_edge_index[0]],p=2,dim=-1) < self.a2a_radius]
        a2a_edge_vector = transform_point_to_local_coordinate(point=a2a_position_a[a2a_edge_index[0]],
                                                              position=a2a_position_a[a2a_edge_index[1]],
                                                              heading=a2a_heading_a[a2a_edge_index[1]])
        a2a_edge_length, a2a_length_theta = compute_angles_lengths_2D(a2a_edge_vector)
        a2a_edge_heading = wrap_angle(a2a_heading_a[a2a_edge_index[0]] - a2a_heading_a[a2a_edge_index[1]])
        a2a_edge_attr_input = torch.stack([a2a_edge_length, a2a_length_theta, a2a_edge_heading], dim=-1)
        a2a_edge_attr_embs = self.a2a_edge_attr_embedding(a2a_edge_attr_input)

        a_embs = a_embs.reshape(-1, self.hidden_dim)
        a_embs_t = self.t2a_graph_attention(x=a_embs, edge_index=t2a_edge_index, edge_attr=t2a_edge_attr_embs)
        a_embs_l = self.l2a_graph_attention(x=[l_embs, a_embs], edge_index=l2a_edge_index, edge_attr=l2a_edge_attr_embs)
        a_embs = a_embs_t + a_embs_l

        a_embs = a_embs.reshape(num_all_agent, self.num_historical_steps, self.hidden_dim)
        for i in range(self.num_layers):
            a_embs = a_embs.transpose(0, 1).reshape(-1, self.hidden_dim)
            a_embs = self.a2a_graph_attention[i](x=a_embs, edge_index=a2a_edge_index, edge_attr=a2a_edge_attr_embs)
            a_embs = a_embs.reshape(self.num_historical_steps, num_all_agent, -1).transpose(0, 1)

        return {
            'a_embs': a_embs,
            'l_embs': l_embs,
        }