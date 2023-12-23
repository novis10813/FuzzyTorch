import torch
import torch.nn as nn
from typing import Union


"""This is an implementation of traditional ANFIS architecture.
FuzzyLayer:
"""


def gaussian_membership_function(x, c, sigma):
    return torch.exp(-(((x - c) / (2 * sigma**2)).pow(2)))


def linear_membership_function(x, c, a):
    return a * (x - c) + 0.5


def bell_shape_membership_function(x, c, a, b=2.0):
    return 1 / (1 + torch.abs((x - c) / a) ** (2 * b))


membership_functions = {
    "gaussian": gaussian_membership_function,
    "bell": bell_shape_membership_function,
    "linear": linear_membership_function,
}


class Fuzzification(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        membership_function: str = "gaussian",
        initializer_centers: Union[torch.Tensor, None] = None,
        initializer_sigmas: Union[torch.Tensor, None] = None,
        **kwargs,
    ):
        super(Fuzzification, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.membership_function = membership_function
        self.output_dim = output_dim
        self.initializer_centers = initializer_centers
        self.initializer_sigmas = initializer_sigmas
        self.input_dim = input_dim
        self.fuzzy_degree = nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        if self.initializer_centers is not None:
            self.fuzzy_degree.data = self.initializer_centers

        self.sigma = nn.Parameter(torch.ones(self.input_dim, self.output_dim))
        if self.initializer_sigmas is not None:
            self.sigma.data = self.initializer_sigmas

        # bell-shape membership function
        if self.membership_function == "bell":
            # set bell-shape shape parameter to 2
            self.b = nn.Parameter(torch.ones(self.input_dim, self.output_dim)*2)
            if self.initializer_shape is not None:
                self.b.data = self.initializer_shape
            
        # 之後可以加入不同的 membership function
        if membership_function in membership_functions:
            self.membership_function = membership_functions[membership_function]
        else:
            raise ValueError(
                f"Membership function {membership_function} is not supported."
            )

    def forward(self, x):
        # x 的 shape [batch_size, input_dim]
        # 最終為 [batch_size, input_dim, output_dim (k)]
        input_variables = []
        # expand x for broadcasting
        batch_size = x.shape[0]
        expanded_x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        input_variables.append(expanded_x)

        # expand fuzzy_degree and sigma for broadcasting
        expanded_fuzzy_degree = self.fuzzy_degree.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, input_dim, output_dim]
        input_variables.append(expanded_fuzzy_degree)
        expanded_sigma = self.sigma.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, input_dim, output_dim]
        input_variables.append(expanded_sigma)

        if self.membership_function == "bell":
            expanded_b = self.b.unsqueeze(0).expand(batch_size, -1, -1)
            input_variables.append(expanded_b)
        
        # calculate fuzzy degree
        fuzzy_out = self.membership_function(*input_variables)
        
        return fuzzy_out


class FuzzyAndRuleLayer(nn.Module):
    def __init__(self):
        super(FuzzyAndRuleLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.min(x, dim=1).values


class FuzzyOrRuleLayer(nn.Module):
    def __init__(self):
        super(FuzzyOrRuleLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=1).values


class SubLayer(nn.Module):
    def __init__(self, output_dim):
        super(SubLayer, self).__init__()
        self.linear = nn.Linear(1, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.linear(x)


class NormalizationLayer(nn.Module):
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sum(x, dim=1, keepdim=True)


class FuzzyLayer(nn.Module):
    def __init__(
        self, input_dim: int, k: int, output_dim: int
    ):
        super(FuzzyLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = k
        self.fuzzification = Fuzzification(
            input_dim=self.input_dim, output_dim=self.output_dim, membership_function="gaussian"
        )
        # self.sublayer = Fuzzification(input_dim=self.input_dim, output_dim=self.output_dim, membership_function="linear")
        self.sublayer = SubLayer(output_dim=self.output_dim)
        self.fuzzy_and_rule_layer = FuzzyAndRuleLayer()
        self.fuzzy_or_rule_layer = FuzzyOrRuleLayer()
        self.normalization_layer = NormalizationLayer()
        self.defuzzification = nn.Linear(self.input_dim * 2, output_dim)

    def forward(self, x):
        # first pass to membership function
        out = self.fuzzification(x)
        sub = self.sublayer(x)
        
        # use both AND and OR rule
        x1 = self.fuzzy_and_rule_layer(out)
        x1 = self.normalization_layer(x1)  # 64, k
        
        x2 = self.fuzzy_or_rule_layer(out)
        x2 = self.normalization_layer(x2)
        

        # defuzzification
        x1_sub = torch.sum(x1.unsqueeze(1) * sub, dim=2) # (batch_size, input_dim)
        x2_sub = torch.sum(x2.unsqueeze(1) * sub, dim=2) # (batch_size, input_dim)
        out = self.defuzzification(torch.cat([x1_sub, x2_sub], dim=1))
        return out.squeeze(1)
