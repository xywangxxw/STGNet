import torch

class DynamicDropEdge:
    def __init__(self, base_distance: float, min_distance: float, max_distance: float):
        self.base = base_distance
        self.min = min_distance
        self.max = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 rel_distance: torch.Tensor,
                 offset: torch.Tensor):

        dynamic_radius = torch.clamp(offset * 10 + self.base, self.min, self.max).squeeze(-1)
        dynamic_radius_edges = dynamic_radius[edge_index[1]]

        mask = rel_distance < dynamic_radius_edges

        new_edge_index = edge_index[:, mask]
        new_edge_attr = edge_attr[mask] if edge_attr is not None else None

        return new_edge_index, new_edge_attr