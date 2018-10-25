import torch
import torch.nn as nn
from torch.nn import functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_size=None, embedding_matrix=None,
                 fine_tune=True, dropout=0.5,
                 padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False,
                 device=torch.device("cpu")):
        '''
        Embedding Layer need at least one of `embedding_size` and `embedding_matrix`
        :param embedding_size: tuple, contains 2 integers indicating the shape of embedding matrix, eg: (20000, 300)
        :param embedding_matrix: torch.Tensor, the pre-trained value of embedding matrix
        :param fine_tune: boolean, whether fine tune embedding matrix
        :param dropout: float, dropout rate
        :param padding_idx: int, if given, pads the output with zeros whenever it encounters the index
        :param max_norm: float, if given, will renormalize the embeddings to always have a norm lesser than this
        :param norm_type: float, the p of the p-norm to compute for the max_norm option
        :param scale_grad_by_freq: boolean, if given, this will scale gradients by the frequency of the words in the mini-batch
        :param sparse: boolean, *unclear option copied from original module*
        '''
        super(EmbeddingLayer, self).__init__()

        if embedding_matrix is not None:
            embedding_size = embedding_matrix.size()
        else:
            embedding_matrix = torch.nn.init.uniform_(torch.FloatTensor(embedding_size[0], embedding_size[1]),
                                                      a=-0.15,
                                                      b=0.15)
        assert (embedding_size is not None)
        assert (embedding_matrix is not None)
        # Config copying
        self.matrix = nn.Embedding(num_embeddings=embedding_size[0],
                                   embedding_dim=embedding_size[1],
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type,
                                   scale_grad_by_freq=scale_grad_by_freq,
                                   sparse=sparse)
        self.matrix.weight.data.copy_(embedding_matrix)
        self.matrix.weight.requires_grad = fine_tune
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None

        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        Forward this module
        :param x: torch.LongTensor, token sequence or sentence, shape is [batch, sentence_len]
        :return: torch.FloatTensor, output data, shape is [batch, sentence_len, embedding_size]
        '''
        if self.dropout is not None:
            return F.dropout(self.matrix(x), p=self.dropout, training=self.training)
        else:
            return self.matrix(x)


class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self, embedding_size=None, embedding_matrix=None,
                 fine_tune=True, dropout=0.5,
                 padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False,
                 device=torch.device("cpu")):
        '''
        MultiLabelEmbeddingLayer Layer need at least one of `embedding_size` and `embedding_matrix`
        :param embedding_size: tuple, contains 2 integers indicating the shape of embedding matrix, eg: (20000, 300)
        :param embedding_matrix: torch.Tensor, the pre-trained value of embedding matrix
        :param fine_tune: boolean, whether fine tune embedding matrix
        :param dropout: float, dropout rate
        :param padding_idx: int, if given, pads the output with zeros whenever it encounters the index
        :param max_norm: float, if given, will renormalize the embeddings to always have a norm lesser than this
        :param norm_type: float, the p of the p-norm to compute for the max_norm option
        :param scale_grad_by_freq: boolean, if given, this will scale gradients by the frequency of the words in the mini-batch
        :param sparse: boolean, *unclear option copied from original module*
        '''
        super(MultiLabelEmbeddingLayer, self).__init__()

        if embedding_matrix is not None:
            embedding_size = embedding_matrix.size()
        else:
            embedding_matrix = torch.randn(embedding_size[0], embedding_size[1])
        assert (embedding_size is not None)
        assert (embedding_matrix is not None)
        # Config copying
        self.matrix = nn.Embedding(num_embeddings=embedding_size[0],
                                   embedding_dim=embedding_size[1],
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type,
                                   scale_grad_by_freq=scale_grad_by_freq,
                                   sparse=sparse)
        self.matrix.weight.data.copy_(embedding_matrix)
        self.matrix.weight.requires_grad = fine_tune
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None

        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        Forward this module
        :param x: list, token sequence or sentence, shape is [batch, sentence_len, variable_size(>=1)]
        :return: torch.FloatTensor, output data, shape is [batch, sentence_len, embedding_size]
        '''
        BATCH = len(x)
        SEQ_LEN = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(BATCH)
             for j in range(SEQ_LEN)]
        x = torch.stack(x).view(BATCH, SEQ_LEN, -1)
        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
