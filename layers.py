import torch
import torch.nn as nn
from typing import List, Union


class FuzzyLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initialiser_centers: Union[float, None] = None,
        initialiser_sigmas: Union[float, None] = None,
    ):
        super(FuzzyLayer, self).__init__()
        self.output_dim = output_dim
        self.initialiser_centers = initialiser_centers
        self.initialiser_sigmas = initialiser_sigmas
        self.input_dim = input_dim
        self.fuzzy_degree = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        if self.initialiser_centers is not None:
            self.fuzzy_degree.data = self.initialiser_centers
        else:
            nn.init.uniform_(self.fuzzy_degree.data)

        self.sigma = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        if self.initialiser_sigmas is not None:
            self.sigma.data = self.initialiser_sigmas
        else:
            nn.init.ones_(self.sigma.data)

    def forward(self, x):
        x = x.unsqueeze(-1).expand(-1, -1, self.output_dim)
        fuzzy_out = torch.exp(
            -torch.sum(((x - self.fuzzy_degree) / (self.sigma**2)).pow(2), dim=-2)
        )
        return fuzzy_out


class FuzzyRuleLayer(nn.Module):
    def __init__(self):
        super(FuzzyRuleLayer, self).__init__()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        out = torch.ones_like(x[0])
        for i in range(len(x)):
            out *= x[i]

        return out


class MFLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        fuzzy_num: int,
        initialiser_centers=None,
        initialiser_sigmas=None,
    ):
        super(MFLayer, self).__init__()
        self.input_dim = input_dim
        self.fuzzy_num = fuzzy_num
        # initialize fuzzy layer for one feature
        self.member_functions = nn.ModuleList(
            [
                FuzzyLayer(
                    1,
                    self.fuzzy_num,
                    initialiser_centers,
                    initialiser_sigmas,
                )
                for _ in range(self.input_dim)
            ]
        )
        self.multiplication_layer = FuzzyRuleLayer()

    def forward(self, x) -> torch.Tensor:
        fuzz = [
            self.member_functions[i](x[:, i].unsqueeze(1))
            for i in range(self.input_dim)
        ]
        x = self.multiplication_layer(fuzz)
        return x


class DenseLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: Union[float, None] = None,
    ):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.dense(x)


class FusionLayer(nn.Module):
    def __init__(
        self, fuzzy_dim: int, dense_dim: int, output_dim, dropout: Union[float, None] = None
    ):
        super(FusionLayer, self).__init__()
        self.fuzzy_dim = fuzzy_dim
        self.dense_dim = dense_dim
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        self.fusion = nn.Linear(self.fuzzy_dim + self.dense_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, dense, fuzzy):
        out = self.fusion(torch.cat([dense, fuzzy], dim=-1))
        out = self.batch_norm(out)
        out = self.dropout(out)
        return torch.relu(out)


class FDNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        fuzzy_num: int,
        hidden_dim: int,
        fusion_out: int,
        output_dim: int,
        dropout: Union[float, None] = None,
        initialiser_centers: Union[float, None] = None,
        initialiser_sigmas: Union[float, None] = None,
    ):
        super(FDNN, self).__init__()
        self.input_dim = input_dim
        self.fuzzy_num = fuzzy_num
        self.hidden_dim = hidden_dim
        self.fusion_out = fusion_out
        self.output_dim = output_dim
        self.mf_layer = MFLayer(
            self.input_dim,
            self.fuzzy_num,
            initialiser_centers,
            initialiser_sigmas,
        )
        self.dense = DenseLayer(
            self.input_dim, self.hidden_dim, self.fuzzy_num, dropout=dropout
        )
        self.fusion = FusionLayer(self.fuzzy_num, self.fuzzy_num, self.fusion_out, dropout=dropout)
        # working layer
        self.output_layer = nn.Linear(self.fusion_out, self.output_dim)

    def forward(self, x):
        # fuzzy layer
        fuzzy = self.mf_layer(x)
        dense = self.dense(x)  # shape (batch_size, fusion_in)
        # fuse fuzzy and dense
        fusion = self.fusion(dense, fuzzy)
        return self.output_layer(fusion).squeeze(1)
