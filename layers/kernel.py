import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.feature_builder import SPDFeatureBuilder

from utils import drop_edge_between_samples

class BaseKernel(nn.Module):
    def forward(self, X, Y):
        raise NotImplementedError

class SPDLinearKernel(BaseKernel):
    def forward(self, X, Y):
        return torch.einsum("mij,nij->mn", X, Y)

class DotProductKernel(BaseKernel):
    def forward(self, X, Y):
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)
        return torch.matmul(X, Y.transpose(0, 1))

class FeatureCorrelationKernel(nn.Module):
    def __init__(
        self,
        in_dim,
        proj_dim,
        kernel_type="spd_linear",
        eps=1e-4
    ):
        super().__init__()

        self.kernel_type = kernel_type

        if kernel_type.startswith("spd"):
            self.spd_builder = SPDFeatureBuilder(
                in_dim=in_dim,
                proj_dim=proj_dim,
                eps=eps
            )

            if kernel_type == "spd_linear":
                self.kernel = SPDLinearKernel()
            else:
                raise NotImplementedError(kernel_type)

        elif kernel_type == "dot":
            self.kernel = DotProductKernel()
        else:
            raise NotImplementedError(kernel_type)

    def forward(self, x, batch=None):
        if self.kernel_type.startswith("spd"):
            spd = self.spd_builder(x)          # [M, D_l, D_l]
            sim = self.kernel(spd, spd)        # [M, M]
        else:
            sim = self.kernel(x, x)            # [M, M]

        if batch is not None:
            sim = drop_edge_between_samples(sim, batch).squeeze(0)

        return sim
