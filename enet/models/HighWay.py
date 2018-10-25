import torch.nn as nn
from torch.nn import functional as F

from enet.util import BottledXavierLinear


class HighWay(nn.Module):
    def __init__(self, size, num_layers=1, dropout_ratio=0.5):
        super(HighWay, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = dropout_ratio

        for i in range(num_layers):
            tmptrans = BottledXavierLinear(size, size)
            tmpgate = BottledXavierLinear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def forward(self, x):
        '''
        forward this module
        :param x: torch.FloatTensor, (N, D) or (N1, N2, D)
        :return: torch.FloatTensor, (N, D) or (N1, N2, D)
        '''

        g = F.sigmoid(self.gate[0](x))
        h = F.relu(self.trans[0](x))
        x = g * h + (1 - g) * x

        for i in range(1, self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            g = F.sigmoid(self.gate[i](x))
            h = F.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x
