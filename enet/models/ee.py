import copy

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from enet import consts
from enet.models.DynamicLSTM import DynamicLSTM
from enet.models.EmbeddingLayer import EmbeddingLayer, MultiLabelEmbeddingLayer
from enet.models.GCN import GraphConvolution
from enet.models.HighWay import HighWay
from enet.models.SelfAttention import AttentionLayer
from enet.models.model import Model
from enet.testing import EDTester
from enet.util import BottledXavierLinear


class EDModel(Model):
    def __init__(self, hyps, device=torch.device("cpu"), embeddingMatrix=None):
        super(EDModel, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.device = device

        # Word Embedding Layer
        self.wembeddings = EmbeddingLayer(embedding_size=(hyps["wemb_size"], hyps["wemb_dim"]),
                                          embedding_matrix=embeddingMatrix,
                                          fine_tune=hyps["wemb_ft"],
                                          dropout=hyps["wemb_dp"],
                                          device=device)

        # Positional Embedding Layer
        self.psembeddings = EmbeddingLayer(embedding_size=(hyps["psemb_size"], hyps["psemb_dim"]),
                                           dropout=hyps["psemb_dp"],
                                           device=device)

        # POS-Tagging Embedding Layer
        self.pembeddings = EmbeddingLayer(embedding_size=(hyps["pemb_size"], hyps["pemb_dim"]),
                                          dropout=hyps["pemb_dp"],
                                          device=device)

        # Entity Label Embedding Layer
        self.eembeddings = MultiLabelEmbeddingLayer(embedding_size=(hyps["eemb_size"], hyps["eemb_dim"]),
                                                    dropout=hyps["eemb_dp"],
                                                    device=device)

        # Bi-LSTM Encoder
        self.bilstm = DynamicLSTM(input_size=hyps["wemb_dim"] +
                                             hyps["psemb_dim"] +
                                             hyps["pemb_dim"] +
                                             hyps["eemb_dim"],
                                  hidden_size=hyps["lstm_dim"],
                                  num_layers=hyps["lstm_layers"],
                                  dropout=hyps["lstm_dp"],
                                  bidirectional=True,
                                  device=device)

        # GCN
        self.gcns = nn.ModuleList()
        for i in range(hyps["gcn_layers"]):
            gcn = GraphConvolution(in_features=2 * hyps["lstm_dim"],
                                   out_features=2 * hyps["lstm_dim"],
                                   edge_types=hyps["gcn_et"],
                                   dropout=hyps["gcn_dp"] if i != hyps["gcn_layers"] - 1 else None,
                                   use_bn=hyps["gcn_use_bn"],
                                   device=device)
            self.gcns.append(gcn)

        # Highway
        if hyps["use_highway"]:
            self.hws = nn.ModuleList()
            for i in range(hyps["gcn_layers"]):
                hw = HighWay(size=2 * hyps["lstm_dim"], dropout_ratio=hyps["gcn_dp"])
                self.hws.append(hw)

        # self attention
        self.sa = AttentionLayer(D=2 * hyps["lstm_dim"], H=hyps["sa_dim"], return_sequences=False)

        # Output Linear
        self.ol = BottledXavierLinear(in_features=2 * hyps["lstm_dim"], out_features=hyps["oc"]).to(device=device)

        # AE Output Linear
        self.ae_ol = BottledXavierLinear(in_features=4 * hyps["lstm_dim"], out_features=hyps["ae_oc"]).to(device=device)

        # Move to right device
        self.to(self.device)

    def get_sentence_positional_feature(self, BATCH_SIZE, SEQ_LEN):
        positions = [[abs(j) for j in range(-i, SEQ_LEN - i)] for i in range(SEQ_LEN)]  # list [SEQ_LEN, SEQ_LEN]
        positions = [torch.LongTensor(position) for position in positions]  # list of tensors [SEQ_LEN]
        positions = [torch.cat([position] * BATCH_SIZE).resize_(BATCH_SIZE, position.size(0))
                     for position in positions]  # list of tensors [BATCH_SIZE, SEQ_LEN]
        return positions

    def forward(self, word_sequence, x_len, pos_tagging_sequence, entity_type_sequence, adj, batch_golden_entities,
                label_i2s):
        '''
        extracting event triggers

        :param word_sequence: LongTensor, padded word indices, (batch_size, seq_len)
        :param lemma_sequence: LongTensor, padded lemma indices, (batch_size, seq_len)
        :param x_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size,)
        :param pos_tagging_sequence: LongTensor, padded pos-tagging label indices, (batch_size, seq_len)
        :param entity_type_sequence: list, padded entity label indices keep all possible labels, (batch_size, seq_len, variable_length>=1)
        :param adj: sparse.FloatTensor, adjacent matrix for provided graph of padded sequences, (batch_size, edge_types, seq_len, seq_len)
        :param batch_golden_entities: [[(st, ed, entity_type_str), ...], ...]
        :param label_i2s:
        :return:
            logits: FloatTensor, output logits of ED, (batch_size, seq_len, output_class)
            mask: ByteTensor, mask of input sequence, (batch_size, seq_len)
            ae_hidden: FloatTensor, output logits of AE, (N, output_class) or [] indicating no need to predicting arguments
            ae_logits_key: [], indicating how the big batch is constructed or []
        '''
        # Merge embeddings
        mask = numpy.zeros(shape=word_sequence.size(), dtype=numpy.uint8)
        for i in range(word_sequence.size()[0]):
            s_len = int(x_len[i])
            mask[i, 0:s_len] = numpy.ones(shape=(s_len), dtype=numpy.uint8)
        mask = torch.ByteTensor(mask).to(self.device)

        word_emb = self.wembeddings(word_sequence)
        pos_emb = self.pembeddings(pos_tagging_sequence)
        entity_label_emb = self.eembeddings(entity_type_sequence)
        x_emb = torch.cat([word_emb, pos_emb, entity_label_emb], 2)  # (batch_size, seq_len, d)

        BATCH_SIZE = word_sequence.size()[0]
        SEQ_LEN = word_sequence.size()[1]
        positional_sequences = self.get_sentence_positional_feature(BATCH_SIZE, SEQ_LEN)
        xx = []
        for i in range(SEQ_LEN):
            # encoding
            x, _ = self.bilstm(torch.cat([x_emb, self.psembeddings(positional_sequences[i].to(self.device))], 2),
                               x_len)  # (batch_size, seq_len, d')
            # gcns
            for i in range(self.hyperparams["gcn_layers"]):
                if self.hyperparams["use_highway"]:
                    x = self.gcns[i](x, adj) + self.hws[i](x)  # (batch_size, seq_len, d')
                else:
                    x = self.gcns[i](x, adj)

            # self attention
            xx.append(self.sa(x, mask))  # (batch_size, d')
        # output linear
        xx = torch.stack(xx, dim=1)  # (batch_size, seq_len, d')
        logits = self.ol(xx)

        ae_logits_key = []
        ae_hidden = []
        trigger_outputs = torch.max(logits, 2)[1].view(logits.size()[:2])  # (batch_size, seq_len)
        for i in range(BATCH_SIZE):
            predicted_event_triggers = EDTester.merge_segments(
                [label_i2s[x] for x in trigger_outputs[i][:x_len[i]].tolist()])
            golden_entities = batch_golden_entities[i]
            golden_entity_tensors = {}
            for j in range(len(golden_entities)):
                e_st, e_ed, e_type_str = golden_entities[j]
                try:
                    golden_entity_tensors[golden_entities[j]] = xx[i, e_st:e_ed, ].mean(dim=0)  # (d')
                except:
                    print(xx.size())
                    print(e_st, e_ed)
                    print(xx[i, e_st:e_ed, ].mean(dim=0).size())
                    exit(-1)

            for st in predicted_event_triggers:
                ed, trigger_type_str = predicted_event_triggers[st]
                event_tensor = xx[i, st:ed, ].mean(dim=0)  # (d')
                for j in range(len(golden_entities)):
                    e_st, e_ed, e_type_str = golden_entities[j]
                    entity_tensor = golden_entity_tensors[golden_entities[j]]
                    ae_hidden.append(torch.cat([event_tensor, entity_tensor]))  # (2 * d')
                    ae_logits_key.append((i, st, ed, trigger_type_str, e_st, e_ed, e_type_str))
        if len(ae_hidden) != 0:
            ae_hidden = self.ae_ol(torch.stack(ae_hidden, dim=0))

        return logits, mask, ae_hidden, ae_logits_key

    def calculate_loss_ed(self, logits, mask, label, weight):
        '''
        Calculate loss for a batched output of ed

        :param logits: FloatTensor, (batch_size, seq_len, output_class)
        :param mask: ByteTensor, mask of padded batched input sequence, (batch_size, seq_len)
        :param label: LongTensor, golden label of paadded sequences, (batch_size, seq_len)
        :return: Float, accumulated loss and index
        '''
        BATCH = logits.size()[0]
        SEQ_LEN = logits.size()[1]
        output = logits.view(BATCH * SEQ_LEN, -1)
        label = label.view(BATCH * SEQ_LEN, -1)
        mask = mask.view(BATCH * SEQ_LEN, )
        masked_index = torch.LongTensor([x for x in range(BATCH * SEQ_LEN) if mask[x] == 1]).to(self.device)
        output_ = output.index_select(0, masked_index)
        label_ = label.index_select(0, masked_index).squeeze(1)
        if weight is not None:
            weight = weight.to(self.device)
            loss = F.nll_loss(F.log_softmax(output_, dim=1), label_, weight=weight)
        else:
            loss = F.nll_loss(F.log_softmax(output_, dim=1), label_)
        return loss

    def calculate_loss_ae(self, logits, keys, batch_golden_events, BATCH_SIZE):
        '''
        Calculate loss for a batched output of ae

        :param logits: FloatTensor, (N, output_class)
        :param keys: [(i, st, ed, trigger_type_str, e_st, e_ed, e_type_str), ...]
        :param batch_golden_events:
        [
            {
                (2, 3, "event_type_str") --> [(1, 2, XX), ...]
                , ...
            }, ...
        ]
        :param BATCH_SIZE: int
        :return:
            loss: Float, accumulated loss and index
            predicted_events:
            [
                {
                    (2, 3, "event_type_str") --> [(1, 2, XX), ...]
                    , ...
                }, ...
            ]
        '''
        # print(batch_golden_events)
        golden_labels = []
        for i, st, ed, event_type_str, e_st, e_ed, entity_type in keys:
            label = consts.ROLE_O_LABEL
            if (st, ed, event_type_str) in batch_golden_events[i]:  # if event matched
                for e_st_, e_ed_, r_label in batch_golden_events[i][(st, ed, event_type_str)]:
                    if e_st == e_st_ and e_ed == e_ed_:
                        label = r_label
                        break
            golden_labels.append(label)
        golden_labels = torch.LongTensor(golden_labels).to(self.device)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), golden_labels)

        predicted_events = [{} for _ in range(BATCH_SIZE)]
        output_ae = torch.max(logits, 1)[1].view(golden_labels.size()).tolist()
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), ae_label in zip(keys, output_ae):
            if ae_label == consts.ROLE_O_LABEL: continue
            if (st, ed, event_type_str) not in predicted_events[i]:
                predicted_events[i][(st, ed, event_type_str)] = []
            predicted_events[i][(st, ed, event_type_str)].append((e_st, e_ed, ae_label))

        return loss, predicted_events
