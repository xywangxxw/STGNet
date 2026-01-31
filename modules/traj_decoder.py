import time
from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import DynamicNeighboringModule
from layers import FeatureCorrelationKernel
from layers import GraphAttention
from layers import MLP
from layers import MutualViewpointPositionalEncoding
from utils import DynamicDropEdge
from utils import compute_angles_lengths_2D
from utils import drop_edge_between_samples
from utils import generate_similarity_edge_index
from utils import init_weights
from utils import transform_point_to_local_coordinate
from utils import transform_traj_to_global_coordinate
from utils import wrap_angle


class TrajDecoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pos_duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 dynbr_psi: float,
                 num_attn_layers: int,
                 num_modes: int,
                 num_heads: int,
                 dropout: float):
        super(TrajDecoder, self).__init__()
        self.q_attn_neighbors = 4
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.dynbr_psi = dynbr_psi
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.num_anchors = num_future_steps // 10
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)

        self.t2m_edge_attr_embedding = MLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2m_edge_attr_embedding = MLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_a_embedding_layer = MLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_s_embedding_layer = MLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.t2m_edge_update_layer = MutualViewpointPositionalEncoding(hidden_dim=hidden_dim, edge_dim=6)
        self.l2m_edge_update_layer = MutualViewpointPositionalEncoding(hidden_dim=hidden_dim)

        self.t2m_graph_attention = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.l2m_graph_attention = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.inDegree_project = nn.Linear(1, hidden_dim)
        self.length_project = nn.Linear(1, hidden_dim)
        self.radius_offset = MLP(input_dim=3 * hidden_dim, hidden_dim=hidden_dim, output_dim=1)

        self.dynbr = DynamicNeighboringModule(
            hidden_dim=self.hidden_dim,
            num_modes=self.num_modes,
            radius=self.a2a_radius,
            embedding_layer=self.m2m_a_embedding_layer,
            graph_attention_layer=GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True),
            in_degree_project=nn.Linear(1, hidden_dim),
            length_project=nn.Linear(1, hidden_dim),
            radius_offset=MLP(input_dim=3 * hidden_dim, hidden_dim=hidden_dim, output_dim=1),
            drop_edge_strategy=DynamicDropEdge(base_distance=a2a_radius, min_distance=a2a_radius, max_distance=self.dynbr_psi*a2a_radius),
        )

        self.m2m_a_graph_attention = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=True,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_edge_update_layer = MutualViewpointPositionalEncoding(hidden_dim=hidden_dim)
        self.m2m_s_graph_attention = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=False,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])
        self.fci_proj_dim = hidden_dim // 4
        self.fci_kernel = FeatureCorrelationKernel(
            in_dim=self.hidden_dim,
            proj_dim=self.fci_proj_dim,
            kernel_type="spd_linear"
        )
        self.m2m_q_graph_attention = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=False,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_propose = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps * 2)
        self.proposal_embedding = MLP(input_dim=(self.num_future_steps // self.num_anchors) * 2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.t2n_edge_update_layer = MutualViewpointPositionalEncoding(hidden_dim=hidden_dim, edge_dim=6)
        self.l2n_edge_update_layer = MutualViewpointPositionalEncoding(hidden_dim=hidden_dim)

        self.t2n_graph_attention_refine = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.l2n_graph_attention_refine = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.n2n_a_graph_attention_refine = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                           dropout=dropout, has_edge_attr=True,
                                                           if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_edge_update_layer = MutualViewpointPositionalEncoding(hidden_dim=hidden_dim)
        self.n2n_s_graph_attention_refine = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                           dropout=dropout, has_edge_attr=True,
                                                           if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_s_edge_update_layer = MutualViewpointPositionalEncoding(hidden_dim=hidden_dim)
        self.n2n_q_graph_attention_refine = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=False,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])


        self.traj_refine = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps // self.num_anchors * 2)

        self.prob_input_proj = nn.Linear(self.num_anchors * self.hidden_dim, self.hidden_dim)
        self.prob_decoder = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.prob_norm = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def forward(self,
                data: HeteroData,
                scene_embed: Dict) -> Dict:

        a_embs, l_embs = scene_embed['a_embs'], scene_embed['l_embs']
        num_all_agent = a_embs.shape[0]

        m_position = data['agent']['position'][:, self.num_historical_steps - 1]
        m_heading = data['agent']['heading'][:, -1]
        m_valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        m_length = data['agent']['length'][:, self.num_historical_steps - 1]
        m_length = m_length.unsqueeze(1).repeat_interleave(self.num_modes, 1)
        m_position = m_position.unsqueeze(1).repeat_interleave(self.num_modes, 1)
        m_heading = m_heading.unsqueeze(1).repeat_interleave(self.num_modes, 1)
        m_valid_mask = m_valid_mask.unsqueeze(1).repeat_interleave(self.num_modes, 1)
        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes, 1)

        t_embs = a_embs.reshape(-1, self.hidden_dim)
        l_embs = l_embs.repeat(self.num_modes, 1)

        m_embs = self.mode_tokens.weight.repeat(num_all_agent, 1)
        t2m_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)
        t2m_heading_t = data['agent']['heading'].reshape(-1)
        t2m_position_m = m_position.reshape(-1, 2)
        t2m_heading_m = m_heading.reshape(-1)
        t2m_valid_mask_t = data['agent']['valid_mask'][:, :self.num_historical_steps]
        t2m_valid_mask_t[:, :self.num_historical_steps - self.pos_duration] = False
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent, -1)
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)
        t2m_edge_index = dense_to_sparse(t2m_valid_mask.contiguous())[0]
        t2m_edge_vector = transform_point_to_local_coordinate(point=t2m_position_t[t2m_edge_index[0]],
                                                              position=t2m_position_m[t2m_edge_index[1]],
                                                              heading=t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] % self.num_historical_steps - self.num_historical_steps
        t2m_edge_attr_input = torch.stack([t2m_edge_attr_length, t2m_edge_attr_theta.sin(), t2m_edge_attr_theta.cos(), t2m_edge_attr_heading.sin(), t2m_edge_attr_heading.cos(), t2m_edge_attr_interval], dim=-1)
        t2m_edge_attr_embs = self.t2m_edge_attr_embedding(t2m_edge_attr_input)
        t2m_edge_attr_embs = self.t2m_edge_update_layer(positions_i=t2m_position_t,
                                                        positions_j=t2m_position_m,
                                                        headings_i=t2m_heading_t,
                                                        headings_j=t2m_heading_m,
                                                        edge_index=t2m_edge_index,
                                                        edge_attr=t2m_edge_attr_embs,
                                                        node_embs_i=t_embs,
                                                        node_embs_j=m_embs,
                                                        time_intervals=t2m_edge_attr_interval)

        l2m_position_l = data['lane']['position']
        l2m_position_m = m_position.reshape(-1, 2)
        l2m_heading_l = data['lane']['heading']
        l2m_heading_m = m_heading.reshape(-1)
        l2m_batch_l = data['lane']['batch']
        l2m_batch_m = m_batch.reshape(-1)
        l2m_valid_mask_l = data['lane']['visible_mask']
        l2m_valid_mask_m = m_valid_mask.reshape(-1)
        l2m_valid_mask = l2m_valid_mask_l.unsqueeze(1) & l2m_valid_mask_m.unsqueeze(0)
        l2m_valid_mask = drop_edge_between_samples(l2m_valid_mask, batch=(l2m_batch_l, l2m_batch_m))
        l2m_edge_index = dense_to_sparse(l2m_valid_mask)[0]
        l2m_edge_vector = transform_point_to_local_coordinate(point=l2m_position_l[l2m_edge_index[0]],
                                                              position=l2m_position_m[l2m_edge_index[1]],
                                                              heading=l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
        l2m_edge_attr_heading = wrap_angle(l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_input = torch.stack([l2m_edge_attr_length, l2m_edge_attr_theta.sin(), l2m_edge_attr_theta.cos(), l2m_edge_attr_heading.sin(), l2m_edge_attr_heading.cos()], dim=-1)
        l2m_edge_attr_embs = self.l2m_edge_attr_embedding(l2m_edge_attr_input)
        l2m_edge_attr_embs = self.l2m_edge_update_layer(positions_i=l2m_position_l,
                                                        positions_j=l2m_position_m,
                                                        headings_i=l2m_heading_l,
                                                        headings_j=l2m_heading_m,
                                                        edge_index=l2m_edge_index,
                                                        edge_attr=l2m_edge_attr_embs,
                                                        node_embs_i=l_embs,
                                                        node_embs_j=m_embs)

        m2m_a_position = m_position.permute(1, 0, 2).reshape(-1, 2)
        m2m_a_heading = m_heading.permute(1, 0).reshape(-1)
        m2m_a_batch = data['agent']['batch']
        m2m_a_valid_mask = m_valid_mask.permute(1, 0)
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(1)
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask.contiguous())[0]
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]]
        m2m_a_edge_index = m2m_a_edge_index[:, torch.norm(m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]], p=2,
                                      dim=-1) < self.a2a_radius]
        m2m_a_edge_vector = transform_point_to_local_coordinate(point=m2m_a_position[m2m_a_edge_index[0]],
                                                                position=m2m_a_position[m2m_a_edge_index[1]],
                                                                heading=m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector)
        m2m_a_edge_attr_heading = wrap_angle(m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_input = torch.stack([m2m_a_edge_attr_length, m2m_a_edge_attr_theta.sin(), m2m_a_edge_attr_theta.cos(), m2m_a_edge_attr_heading.sin(), m2m_a_edge_attr_heading.cos()],
                                            dim=-1)
        m2m_a_edge_attr_embs = self.m2m_a_embedding_layer(m2m_a_edge_attr_input)
        m2m_a_edge_attr_embs = self.m2m_a_edge_update_layer(positions_i=m2m_a_position,
                                                            positions_j=m2m_a_position,
                                                            headings_i=m2m_a_heading,
                                                            headings_j=m2m_a_heading,
                                                            edge_index=m2m_a_edge_index, edge_attr=m2m_a_edge_attr_embs,
                                                            node_embs_i=m_embs, node_embs_j=m_embs)

        m2m_s_valid_mask = m_valid_mask
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask.contiguous())[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]

        m_embs_l = self.l2m_graph_attention(x=[l_embs, m_embs], edge_index=l2m_edge_index, edge_attr=l2m_edge_attr_embs)
        m_embs_t = self.t2m_graph_attention(x=[t_embs, m_embs], edge_index=t2m_edge_index, edge_attr=t2m_edge_attr_embs)
        m_embs = m_embs_t + m_embs_l

        m2m_a_edge_index, m2m_a_edge_attr_embs = self.dynbr(
            node_embs=m_embs,
            edge_index=m2m_a_edge_index,
            edge_attr=m2m_a_edge_attr_embs,
            positions=m2m_a_position,
            headings=m2m_a_heading,
            node_lengths=m_length,
            num_all_agent=num_all_agent,
            num_nodes=data['agent']['num_nodes'],
        )

        for i in range(self.num_attn_layers):
            m_embs = m_embs.reshape(num_all_agent, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m_embs = self.m2m_a_graph_attention[i](x=m_embs, edge_index=m2m_a_edge_index, edge_attr=m2m_a_edge_attr_embs)
            m2m_a_edge_attr_embs = self.m2m_a_edge_update_layer(positions_i=m2m_a_position,
                                                                positions_j=m2m_a_position,
                                                                headings_i=m2m_a_heading,
                                                                headings_j=m2m_a_heading,
                                                                edge_index=m2m_a_edge_index, edge_attr=m2m_a_edge_attr_embs, node_embs_i=m_embs, node_embs_j=m_embs)
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m_embs = self.m2m_s_graph_attention[i](x=m_embs, edge_index=m2m_s_edge_index)

        m2m_q_position = m_position.reshape(-1, 2)
        m2m_q_heading = m_heading.reshape(-1)
        m2m_q_batch = m_batch.reshape(-1)
        m2m_q_matrix = self.fci_kernel(
            m_embs,
            batch=m2m_q_batch
        )
        m2m_q_matrix = drop_edge_between_samples(m2m_q_matrix, m2m_q_batch).squeeze(0)
        m2m_q_matrix = generate_similarity_edge_index(m2m_q_matrix, self.q_attn_neighbors)
        m2m_q_edge_index = dense_to_sparse(m2m_q_matrix.contiguous())[0]
        m2m_q_edge_vector = transform_point_to_local_coordinate(point=m2m_q_position[m2m_q_edge_index[0]],
                                                                position=m2m_q_position[m2m_q_edge_index[1]],
                                                                heading=m2m_q_heading[m2m_q_edge_index[1]])
        m2m_q_edge_attr_length, m2m_q_edge_attr_theta = compute_angles_lengths_2D(m2m_q_edge_vector)
        m2m_q_edge_attr_heading = wrap_angle(m2m_q_heading[m2m_q_edge_index[0]] - m2m_q_heading[m2m_q_edge_index[1]])
        m2m_q_edge_attr_input = torch.stack([m2m_q_edge_attr_length, m2m_q_edge_attr_theta.sin(), m2m_q_edge_attr_theta.cos(),
                                             m2m_q_edge_attr_heading.sin(), m2m_q_edge_attr_heading.cos()], dim=-1)
        m2m_q_edge_attr_embs = self.m2m_a_embedding_layer(m2m_q_edge_attr_input)
        m2m_q_edge_attr_embs = self.m2m_a_edge_update_layer(positions_i=m2m_q_position,
                                                            positions_j=m2m_q_position,
                                                            headings_i=m2m_q_heading,
                                                            headings_j=m2m_q_heading,
                                                            edge_index=m2m_q_edge_index, edge_attr=m2m_q_edge_attr_embs,
                                                            node_embs_i=m_embs, node_embs_j=m_embs)

        for i in range(self.num_attn_layers):
            m_embs = self.m2m_q_graph_attention[i](x=m_embs, edge_index=m2m_q_edge_index, edge_attr=m2m_q_edge_attr_embs)
            m2m_q_edge_attr_embs = self.m2m_a_edge_update_layer(positions_i=m2m_q_position,
                                                                positions_j=m2m_q_position,
                                                                headings_i=m2m_q_heading,
                                                                headings_j=m2m_q_heading,
                                                                edge_index=m2m_q_edge_index, edge_attr=m2m_q_edge_attr_embs, node_embs_i=m_embs, node_embs_j=m_embs)

        traj_propose = self.traj_propose(m_embs).reshape(num_all_agent, self.num_modes, self.num_future_steps, 2)
        proposal = traj_propose.detach()

        global_traj_proposal = transform_traj_to_global_coordinate(proposal, m_position, m_heading)

        n_position = torch.zeros(num_all_agent, self.num_modes, self.num_anchors, 2, device=m_embs.device)
        n_length = torch.zeros(num_all_agent, self.num_modes, self.num_anchors, device=m_embs.device)
        n_heading = torch.zeros(num_all_agent, self.num_modes, self.num_anchors, device=m_embs.device)
        n_embs = torch.zeros(num_all_agent, self.num_modes, self.num_anchors, self.hidden_dim, device=m_embs.device)

        for i in range(self.num_anchors):
            mid = (2 * i + 1) * 10 // 2
            n_position[:, :, i, :] = global_traj_proposal[:, :, mid, :]
            n_length[:, :, i], n_heading[:, :, i] = compute_angles_lengths_2D(global_traj_proposal[:, :, mid, :] - global_traj_proposal[:, :, mid - 1, :])
            partial_proposal = proposal[:, :, i * 10 : (i + 1) * 10, :]
            n_embs[:, :, i, :] = self.proposal_embedding(partial_proposal.reshape(-1, self.num_future_steps // self.num_anchors * 2)).reshape(num_all_agent, self.num_modes, self.hidden_dim)
        n_batch = m_batch.unsqueeze(-1).repeat_interleave(self.num_anchors, 1)
        n_valid_mask = m_valid_mask.unsqueeze(-1).repeat_interleave(self.num_anchors, 1)

        n_embs = n_embs.reshape(-1, self.hidden_dim)

        t2n_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)
        t2n_heading_t = data['agent']['heading'].reshape(-1)
        t2n_position_n = n_position.reshape(-1, 2)
        t2n_heading_n = n_heading.reshape(-1)
        t2n_valid_mask_t = data['agent']['valid_mask'][:, :self.num_historical_steps]
        t2n_valid_mask_t[:, :self.num_historical_steps - self.pos_duration] = False
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent, -1)
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)
        t2n_edge_index = dense_to_sparse(t2n_valid_mask.contiguous())[0]
        t2n_edge_vector = transform_point_to_local_coordinate(point=t2n_position_t[t2n_edge_index[0]],
                                                              position=t2n_position_n[t2n_edge_index[1]],
                                                              heading=t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]])
        t2n_anchor_index = t2n_edge_index[1] % self.num_anchors
        t2n_edge_attr_interval = t2n_edge_index[0] % self.num_historical_steps - self.num_historical_steps - (10 * t2n_anchor_index + 5)
        t2n_edge_attr_input = torch.stack([t2n_edge_attr_length, t2n_edge_attr_theta.sin(), t2n_edge_attr_theta.cos(), t2n_edge_attr_heading.sin(), t2n_edge_attr_heading.cos(), t2n_edge_attr_interval], dim=-1)
        t2n_edge_attr_embs = self.t2m_edge_attr_embedding(t2n_edge_attr_input)
        t2n_edge_attr_embs = self.t2n_edge_update_layer(positions_i=t2n_position_t,
                                                        positions_j=t2n_position_n,
                                                        headings_i=t2n_heading_t,
                                                        headings_j=t2n_heading_n,
                                                        edge_index=t2n_edge_index,
                                                        edge_attr=t2n_edge_attr_embs,
                                                        node_embs_i=t_embs,
                                                        node_embs_j=n_embs,
                                                        time_intervals=t2n_edge_attr_interval)

        l2n_position_l = data['lane']['position']
        l2n_position_n = n_position.reshape(-1, 2)
        l2n_heading_l = data['lane']['heading']
        l2n_heading_n = n_heading.reshape(-1)
        l2n_batch_l = data['lane']['batch']
        l2n_batch_n = n_batch.reshape(-1)
        l2n_valid_mask_l = data['lane']['visible_mask']
        l2n_valid_mask_n = n_valid_mask.reshape(-1)
        l2n_valid_mask = l2n_valid_mask_l.unsqueeze(1) & l2n_valid_mask_n.unsqueeze(0)
        l2n_valid_mask = drop_edge_between_samples(l2n_valid_mask, batch=(l2n_batch_l, l2n_batch_n))
        l2n_edge_index = dense_to_sparse(l2n_valid_mask.contiguous())[0]
        l2n_edge_index = l2n_edge_index[:, torch.norm(l2n_position_l[l2n_edge_index[0]] - l2n_position_n[l2n_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
        l2n_edge_vector = transform_point_to_local_coordinate(point=l2n_position_l[l2n_edge_index[0]],
                                                              position=l2n_position_n[l2n_edge_index[1]],
                                                              heading=l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
        l2n_edge_attr_heading = wrap_angle(l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_input = torch.stack([l2n_edge_attr_length, l2n_edge_attr_theta.sin(), l2n_edge_attr_theta.cos(), l2n_edge_attr_heading.sin(), l2n_edge_attr_heading.cos()], dim=-1)
        l2n_edge_attr_embs = self.l2m_edge_attr_embedding(l2n_edge_attr_input)
        l2n_edge_attr_embs = self.l2n_edge_update_layer(positions_i=l2n_position_l,
                                                        positions_j=l2n_position_n,
                                                        headings_i=l2n_heading_l,
                                                        headings_j=l2n_heading_n,
                                                        edge_index=l2n_edge_index,
                                                        edge_attr=l2n_edge_attr_embs,
                                                        node_embs_i=l_embs,
                                                        node_embs_j=n_embs)

        n2n_a_position = n_position.permute(1, 2, 0, 3).reshape(-1, 2)
        n2n_a_heading = n_heading.permute(1, 2, 0).reshape(-1)
        n2n_a_batch = data['agent']['batch']
        n2n_a_valid_mask = n_valid_mask.permute(1, 2, 0).reshape(-1, num_all_agent)
        n2n_a_valid_mask = n2n_a_valid_mask.unsqueeze(2) & n2n_a_valid_mask.unsqueeze(1)
        n2n_a_valid_mask = drop_edge_between_samples(n2n_a_valid_mask, n2n_a_batch)
        n2n_a_edge_index = dense_to_sparse(n2n_a_valid_mask.contiguous())[0]
        n2n_a_edge_index = n2n_a_edge_index[:, n2n_a_edge_index[1] != n2n_a_edge_index[0]]
        n2n_a_edge_index = n2n_a_edge_index[:, torch.norm(n2n_a_position[n2n_a_edge_index[1]] - n2n_a_position[n2n_a_edge_index[0]], p=2, dim=-1) < self.a2a_radius]
        n2n_a_edge_vector = transform_point_to_local_coordinate(point=n2n_a_position[n2n_a_edge_index[0]],
                                                                position=n2n_a_position[n2n_a_edge_index[1]],
                                                                heading=n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_length, n2n_a_edge_attr_theta = compute_angles_lengths_2D(n2n_a_edge_vector)
        n2n_a_edge_attr_heading = wrap_angle(n2n_a_heading[n2n_a_edge_index[0]] - n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_input = torch.stack([n2n_a_edge_attr_length, n2n_a_edge_attr_theta.sin(), n2n_a_edge_attr_theta.cos(), n2n_a_edge_attr_heading.sin(), n2n_a_edge_attr_heading.cos()], dim=-1)
        n2n_a_edge_attr_embs = self.m2m_a_embedding_layer(n2n_a_edge_attr_input)
        n2n_a_edge_attr_embs = self.n2n_a_edge_update_layer(positions_i=n2n_a_position,
                                                            positions_j=n2n_a_position,
                                                            headings_i=n2n_a_heading,
                                                            headings_j=n2n_a_heading,
                                                            edge_index=n2n_a_edge_index, edge_attr=n2n_a_edge_attr_embs,
                                                            node_embs_i=n_embs, node_embs_j=n_embs)

        n2n_s_position = n_position.permute(0, 2, 1, 3).reshape(-1, 2)
        n2n_s_heading = n_heading.permute(0, 2, 1).reshape(-1)
        n2n_s_valid_mask = n_valid_mask.permute(0, 2, 1).reshape(-1, self.num_modes)
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask.contiguous())[0]
        n2n_s_edge_vector = transform_point_to_local_coordinate(point=n2n_s_position[n2n_s_edge_index[0]],
                                                                position=n2n_s_position[n2n_s_edge_index[1]],
                                                                heading=n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_length, n2n_s_edge_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_heading = wrap_angle(n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_input = torch.stack([n2n_s_edge_length, n2n_s_edge_theta.sin(), n2n_s_edge_theta.cos(), n2n_s_edge_heading.sin(), n2n_s_edge_heading.cos()], dim=-1)
        n2n_s_edge_attr_embs = self.m2m_s_embedding_layer(n2n_s_edge_attr_input)
        n2n_s_edge_attr_embs = self.n2n_s_edge_update_layer(positions_i=n2n_s_position,
                                                            positions_j=n2n_s_position,
                                                            headings_i=n2n_s_heading,
                                                            headings_j=n2n_s_heading,
                                                            edge_index=n2n_s_edge_index, edge_attr=n2n_s_edge_attr_embs,
                                                            node_embs_i=n_embs, node_embs_j=n_embs)

        n_embs_t = self.t2n_graph_attention_refine(x=[t_embs, n_embs], edge_index=t2n_edge_index, edge_attr=t2n_edge_attr_embs)
        n_embs_l = self.l2n_graph_attention_refine(x=[l_embs, n_embs], edge_index=l2n_edge_index, edge_attr=l2n_edge_attr_embs)
        n_embs = n_embs_t + n_embs_l

        for i in range(self.num_attn_layers):
            n_embs = n_embs.reshape(num_all_agent, self.num_modes, self.num_anchors, self.hidden_dim).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim) # [K*A*(N), D]
            n_embs = self.n2n_a_graph_attention_refine[i](x=n_embs, edge_index=n2n_a_edge_index, edge_attr=n2n_a_edge_attr_embs)
            n2n_a_edge_attr_embs = self.n2n_a_edge_update_layer(positions_i=n2n_a_position,
                                                                positions_j=n2n_a_position,
                                                                headings_i=n2n_a_heading,
                                                                headings_j=n2n_a_heading,
                                                                edge_index=n2n_a_edge_index, edge_attr=n2n_a_edge_attr_embs, node_embs_i=n_embs, node_embs_j=n_embs)
            n_embs = n_embs.reshape(self.num_modes, self.num_anchors, num_all_agent, self.hidden_dim).permute(2, 1, 0, 3).reshape(-1, self.hidden_dim) # [(N)*A*K, D]
            n_embs = self.n2n_s_graph_attention_refine[i](x=n_embs, edge_index=n2n_s_edge_index, edge_attr=n2n_s_edge_attr_embs)
            n2n_s_edge_attr_embs = self.n2n_s_edge_update_layer(positions_i=n2n_s_position,
                                                                positions_j=n2n_s_position,
                                                                headings_i=n2n_s_heading,
                                                                headings_j=n2n_s_heading,
                                                                edge_index=n2n_s_edge_index, edge_attr=n2n_s_edge_attr_embs, node_embs_i=n_embs, node_embs_j=n_embs)

        n_embs = n_embs.reshape(num_all_agent, self.num_anchors, self.num_modes, self.hidden_dim).permute(0, 2, 1, 3)
        n_embs = n_embs.reshape(-1, self.hidden_dim)
        n2n_q_position = n_position.reshape(-1, 2)
        n2n_q_heading = n_heading.reshape(-1)
        n2n_q_batch = n_batch.reshape(-1)
        n2n_q_matrix = self.fci_kernel(
            n_embs,
            n2n_q_batch
        )
        n2n_q_matrix = drop_edge_between_samples(n2n_q_matrix, n2n_q_batch).squeeze(0)
        n2n_q_matrix = generate_similarity_edge_index(n2n_q_matrix, self.q_attn_neighbors)
        n2n_q_edge_index = dense_to_sparse(n2n_q_matrix.contiguous())[0]
        n2n_q_edge_vector = transform_point_to_local_coordinate(point=n2n_q_position[n2n_q_edge_index[0]],
                                                                position=n2n_q_position[n2n_q_edge_index[1]],
                                                                heading=n2n_q_heading[n2n_q_edge_index[1]])
        n2n_q_edge_attr_length, n2n_q_edge_attr_theta = compute_angles_lengths_2D(n2n_q_edge_vector)
        n2n_q_edge_attr_heading = wrap_angle(n2n_q_heading[n2n_q_edge_index[0]] - n2n_q_heading[n2n_q_edge_index[1]])
        n2n_q_edge_attr_input = torch.stack([n2n_q_edge_attr_length, n2n_q_edge_attr_theta.sin(), n2n_q_edge_attr_theta.cos(),
                                             n2n_q_edge_attr_heading.sin(), n2n_q_edge_attr_heading.cos()], dim=-1)
        n2n_q_edge_attr_embs = self.m2m_a_embedding_layer(n2n_q_edge_attr_input)
        n2n_q_edge_attr_embs = self.n2n_a_edge_update_layer(positions_i=n2n_q_position,
                                                            positions_j=n2n_q_position,
                                                            headings_i=n2n_q_heading,
                                                            headings_j=n2n_q_heading,
                                                            edge_index=n2n_q_edge_index, edge_attr=n2n_q_edge_attr_embs,
                                                            node_embs_i=n_embs, node_embs_j=n_embs)

        for i in range(self.num_attn_layers):
            n_embs = self.n2n_q_graph_attention_refine[i](x=n_embs, edge_index=n2n_q_edge_index, edge_attr=n2n_q_edge_attr_embs)
            n2n_q_edge_attr_embs = self.n2n_a_edge_update_layer(positions_i=n2n_q_position,
                                                                positions_j=n2n_q_position,
                                                                headings_i=n2n_q_heading,
                                                                headings_j=n2n_q_heading,
                                                                edge_index=n2n_q_edge_index, edge_attr=n2n_q_edge_attr_embs, node_embs_i=n_embs, node_embs_j=n_embs)

        n_embs = n_embs.reshape(num_all_agent, self.num_modes, self.num_anchors, self.hidden_dim)
        offset = torch.zeros(num_all_agent, self.num_modes, self.num_future_steps, 2, device=n_embs.device)
        for i in range(self.num_anchors):
            offset[:, :, i * 10 : (i + 1) * 10, :] = self.traj_refine(n_embs[:, :, i, :].reshape(-1, self.hidden_dim)).reshape(num_all_agent, self.num_modes, self.num_future_steps // self.num_anchors, 2)

        traj_output = traj_propose + offset

        n_embs = n_embs.reshape(-1, self.num_anchors * self.hidden_dim)
        n_embs = self.prob_input_proj(n_embs)
        prob_output = self.prob_decoder(n_embs.detach()).reshape(-1, self.num_modes)
        prob_output = self.prob_norm(prob_output)


        return {
            'traj_propose': traj_propose,
            'traj_refine': traj_output,
            'prob': prob_output,
        }