import torch
import torch.nn as nn
import torch.nn.functional as F

class SPDFeatureBuilder(nn.Module):
    def __init__(self, in_dim, proj_dim, eps=1e-4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        self.eps = eps

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)

        f = self.proj(x)

        spd = torch.einsum("md,me->mde", f, f)

        eye = torch.eye(spd.size(-1), device=spd.device)
        spd = spd + self.eps * eye.unsqueeze(0)

        return spd

