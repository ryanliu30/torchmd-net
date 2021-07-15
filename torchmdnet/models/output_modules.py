from abc import abstractmethod, ABCMeta
import ase
from torchmdnet.models.utils import act_class_mapping, GatedEquivariantBlock
from torch_scatter import scatter
import torch
from torch import nn


__all__ = ['Scalar', 'DipoleMoment', 'ElectronicSpatialExtent']


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(self, is_equivariant, hidden_channels, activation='silu',
                 allow_prior_model=True):
        super(Scalar, self).__init__(allow_prior_model=allow_prior_model)
        self.is_equivariant = is_equivariant

        if is_equivariant:
            self.output_network = nn.ModuleList([
                GatedEquivariantBlock(hidden_channels, hidden_channels // 2,
                                      activation=activation, scalar_activation=True),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ])
        else:
            act_class = act_class_mapping[activation]
            self.output_network = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                act_class(),
                nn.Linear(hidden_channels // 2, 1)
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.is_equivariant:
            for layer in self.output_network:
                layer.reset_parameters()
        else:
            nn.init.xavier_uniform_(self.output_network[0].weight)
            self.output_network[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.output_network[2].weight)
            self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v, z, pos, batch):
        if self.is_equivariant:
            for layer in self.output_network:
                x, v = layer(x, v)
            # include v in output to make sure all parameters have a gradient
            return x + v.sum() * 0
        return self.output_network(x)


class DipoleMoment(Scalar):
    def __init__(self, is_equivariant, hidden_channels, activation='silu'):
        super(DipoleMoment, self).__init__(is_equivariant, hidden_channels,
                                           activation, allow_prior_model=False)

        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer('atomic_mass', atomic_mass)

    def pre_reduce(self, x, v, z, pos, batch):
        if self.is_equivariant:
            for layer in self.output_network:
                x, v = layer(x, v)
        else:
            x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])

        if self.is_equivariant:
            x = x + v.squeeze()
        return x

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class ElectronicSpatialExtent(OutputModel):
    def __init__(self, is_equivariant, hidden_channels, activation='silu'):
        super(ElectronicSpatialExtent, self).__init__(allow_prior_model=False)
        self.is_equivariant = is_equivariant

        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v, z, pos, batch):
        x = self.output_network(x)
        x = pos.norm(dim=1, keepdim=True) ** 2 * x
        return x
