import torch
from torch import nn
import torch.nn.functional as F


class AggregatorMixin():
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)


class MeanAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, dropout=0.5,
                 combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
        self.dropout = dropout

    def forward(self, x, neibs):
        agg_neib = neibs.view(x.shape[0], -1, neibs.shape[1])
        agg_neib = agg_neib.mean(dim=1)
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        out = F.dropout(out, self.dropout, training=self.training)
        if self.activation:
            out = self.activation(out)

        return out
        
        
aggregator_lookup = {
    "mean": MeanAggregator,
}
