"""Training-only dropout, exactly the existing predictor function during eval."""
import torch
from .lp_validation_repair import LayerNormLinkHead


class RegularizedLinkHead(LayerNormLinkHead):
    def __init__(self, width=1024, dropout=.3):
        super().__init__(width)
        self.training_dropout = dropout

    def forward(self, source, target):
        value = source*target
        for index, layer in enumerate(self.network):
            value = layer(value)
            if index in (2, 5):
                value = torch.nn.functional.dropout(value, p=self.training_dropout, training=self.training)
        return value.squeeze(-1)
