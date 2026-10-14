"""One evidence-conditioned residual network shared by all current partitions."""
import math
import numpy as np
import torch
from torch import nn


class Interaction(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(width) for _ in range(3))
        self.cells = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.evidence = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.ffn = nn.Sequential(nn.Linear(width, 4*width), nn.SiLU(), nn.Linear(4*width, width))

    def forward(self, tokens, evidence):
        z = self.norm1(tokens)
        tokens = tokens + self.cells(z, z, z, need_weights=False)[0]
        tokens = tokens + self.evidence(self.norm2(tokens), evidence, evidence, need_weights=False)[0]
        return tokens + self.ffn(self.norm3(tokens))


class SharedFlux(nn.Module):
    """r_B = q_B + m_B h_B with bounded h and full-state token interactions.

    Width, layers, heads and feature transforms are explicit implementation
    choices, not architectural values supplied by the manuscript.
    """
    def __init__(self, grid, config):
        super().__init__()
        self.grid, self.config = grid, dict(config)
        self.kappa, self.sigma = float(config['kappa']), float(config['sigma'])
        width, heads, blocks = (int(config[k]) for k in ('width', 'heads', 'blocks'))
        self.bound = float(config['residual_bound'])
        rate = 2*self.kappa/self.sigma**2
        if width < 1 or heads < 1 or blocks < 1 or width % heads or self.kappa <= 0 or self.sigma <= 0 or self.bound <= rate:
            raise ValueError('Invalid network/CIR configuration; residual bound must exceed the stationary rate.')
        total = float(config['reference_total_mass'])
        if total <= 0:
            raise ValueError('Reference total mass must be positive.')
        a = grid.areas/grid.areas.sum()*total
        self.register_buffer('reference_mass', torch.tensor(a, dtype=torch.float64))
        self.register_buffer('time_frequencies', 2.0**torch.arange(4)*math.pi)
        # Count prevents normalized attention from erasing the distinction
        # between one acquisition and independently repeated identical rows.
        self.observation_encoder = nn.Sequential(nn.Linear(grid.size+2, width), nn.SiLU(), nn.Linear(width, width), nn.LayerNorm(width))
        # amplitude, reference mass, four bounds, raw time, four sin/cos pairs
        self.cell_encoder = nn.Sequential(nn.Linear(15, width), nn.SiLU(), nn.Linear(width, width))
        self.interactions = nn.ModuleList([Interaction(width, heads) for _ in range(blocks)])
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, math.atanh(-rate/self.bound))

    @property
    def device(self):
        return self.head.weight.device

    @property
    def dtype(self):
        return self.head.weight.dtype

    def encode_evidence(self, evidence):
        if evidence.kernel.shape[1] != self.grid.size:
            raise ValueError('The evidence kernel must remain on the fixed reference grid.')
        raw = np.column_stack((evidence.white_y, evidence.white_kernel))
        values = torch.as_tensor(raw, dtype=self.dtype, device=self.device)
        count = values.new_full((len(values), 1), math.log1p(len(values)))
        return self.observation_encoder(torch.cat((torch.asinh(values), count), dim=-1))[None]

    def partition_reference(self, partition):
        return partition.aggregate(self.reference_mass.detach().cpu().numpy())

    def residual(self, mass, partition, time, encoded):
        x = torch.as_tensor(mass, dtype=self.dtype, device=self.device)
        if x.ndim != 2 or x.shape[1] != partition.size:
            raise ValueError('Mass must have shape [batch,current cells].')
        batch, cells = x.shape
        a = torch.as_tensor(self.partition_reference(partition), dtype=x.dtype, device=x.device)
        bounds = partition.bounds.copy()
        bounds[:, :2] = (bounds[:, :2]-self.grid.t1[0])/(self.grid.t1[-1]-self.grid.t1[0])
        bounds[:, 2:] = (bounds[:, 2:]-self.grid.t2[0])/(self.grid.t2[-1]-self.grid.t2[0])
        b = torch.as_tensor(bounds, dtype=x.dtype, device=x.device).expand(batch, -1, -1)
        t = torch.as_tensor(time, dtype=x.dtype, device=x.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if t.numel() != batch:
            raise ValueError('Supply one time or one time per batch element.')
        t = t/float(self.config['terminal_time'])
        phase = t[:, None]*self.time_frequencies.to(dtype=x.dtype)
        time_features = torch.cat((t[:, None], phase.sin(), phase.cos()), dim=1)
        features = torch.cat((torch.log1p(x/a)[..., None], a.log()[None, :, None].expand(batch, -1, -1),
                              b, time_features[:, None, :].expand(-1, cells, -1)), dim=-1)
        tokens = self.cell_encoder(features)
        if encoded.shape[0] not in (1, batch):
            raise ValueError('Encoded evidence batch is incompatible.')
        for block in self.interactions:
            tokens = block(tokens, encoded.expand(batch, -1, -1))
        return self.bound*torch.tanh(self.head(self.output_norm(tokens)).squeeze(-1))

    def forward(self, mass, partition, time, encoded):
        x = torch.as_tensor(mass, dtype=self.dtype, device=self.device)
        a = torch.as_tensor(self.partition_reference(partition), dtype=x.dtype, device=x.device)
        return 2*self.kappa*a/self.sigma**2 + x*self.residual(x, partition, time, encoded)
