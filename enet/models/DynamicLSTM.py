import numpy
import numpy as np
import torch
import torch.nn as nn


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, device=torch.device("cpu")):
        """
        Dynamic LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.device = device
        self.to(device)

    def forward(self, x, x_len, only_use_last_hidden_state=False):
        """
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort

        :param x: FloatTensor, pre-padded input sequence (batch_size, seq_len, feature_dim)
        :param x_len: numpy list, indicating corresponding actual sequence length
        :return: output, (h_n, c_n)
        - **output**: FloatTensor, packed output sequence (batch_size, seq_len, feature_dim * num_directions)
            containing the output features `(h_t)` from the last layer of the LSTM, for each t.
        - **h_n**: FloatTensor, (num_layers * num_directions, batch, hidden_size)
            containing the hidden state for `t = seq_len`
        - **c_n**: FloatTensor, (num_layers * num_directions, batch, hidden_size)
            containing the cell state for `t = seq_len`
        """
        # 1. sort
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        # 2. pack
        x_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        # 3. process using RNN
        out_pack, (ht, ct) = self.LSTM(x_p, None)
        # 4. unsort h
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        if only_use_last_hidden_state:
            return ht
        else:
            # 5. unpack output
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            # 6. unsort out c
            out = out[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)


if __name__ == "__main__":
    BATCH_SIZE = 5
    SEQ_LEN = 3
    D = 2
    aa = DynamicLSTM(input_size=2, hidden_size=2, batch_first=True)
    binp = torch.rand(BATCH_SIZE, SEQ_LEN, D)
    binlen = numpy.array([3, 3, 3, 2, 2], dtype=numpy.int8)
    boup, _ = aa(binp, binlen)
    print(boup)

    for i in range(BATCH_SIZE):
        sinp = binp[i].unsqueeze(0)
        sinlen = numpy.array([binlen[i]], dtype=numpy.int8)
        soup, _ = aa(sinp, sinlen)
        print(soup)
