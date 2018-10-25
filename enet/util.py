import json
import math
import sys

import torch
import torch.nn as nn
from torch.nn import init

from enet.corpus.Sentence import Token


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OrthogonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class BottledLinear(Bottle, nn.Linear):
    pass


class BottledXavierLinear(Bottle, XavierLinear):
    pass


class BottledOrthogonalLinear(Bottle, OrthogonalLinear):
    pass


def log(*args, **kwargs):
    print(file=sys.stdout, flush=True, *args, **kwargs)


def logerr(*args, **kwargs):
    print(file=sys.stderr, flush=True, *args, **kwargs)


def logonfile(fp, *args, **kwargs):
    fp.write(*args, **kwargs)


def progressbar(cur, total, other_information):
    percent = '{:.2%}'.format(cur / total)
    if type(other_information) is str:
        log("\r[%-50s] %s %s" % ('=' * int(math.floor(cur * 50 / total)), percent, other_information))
    else:
        log("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent))


def save_hyps(hyps, fp):
    json.dump(hyps, fp)


def load_hyps(fp):
    hyps = json.load(fp)
    return hyps


def add_tokens(words, y, y_, x_len, all_tokens, word_i2s, label_i2s):
    words = words.tolist()
    for s, ys, ys_, sl in zip(words, y, y_, x_len):
        s = s[:sl]
        ys = ys[:sl]
        ys_ = ys_[:sl]
        tokens = []
        for w, yw, yw_ in zip(s, ys, ys_):
            atoken = Token(word=word_i2s[w], posLabel="", entityLabel="", triggerLabel=label_i2s[yw], lemmaLabel="")
            atoken.addPredictedLabel(label_i2s[yw_])
            tokens.append(atoken)
        all_tokens.append(tokens)


def run_over_data(model, optimizer, data_iter, MAX_STEP, need_backward, tester, hyps, device, word_i2s, label_i2s,
                  role_i2s, maxnorm, weight, save_output):
    if need_backward:
        model.test_mode_off()
    else:
        model.test_mode_on()

    running_loss = 0.0

    print()

    all_tokens = []
    all_y = []
    all_y_ = []
    all_events = []
    all_events_ = []

    cnt = 0
    for batch in data_iter:
        if need_backward:
            optimizer.zero_grad()

        words, x_len = batch.WORDS
        # lemmas, _ = batch.LEMMAS
        postags = batch.POSTAGS
        entitylabels = batch.ENTITYLABELS
        adjm = batch.ADJM
        y = batch.LABEL
        entities = batch.ENTITIES
        events = batch.EVENT
        all_events.extend(events)

        SEQ_LEN = words.size()[1]
        adjm = torch.stack([torch.sparse.FloatTensor(torch.LongTensor(adjmm[0]),
                                                     torch.FloatTensor(adjmm[1]),
                                                     torch.Size([hyps["gcn_et"], SEQ_LEN, SEQ_LEN])).to_dense() for
                            adjmm in adjm])
        words = words.to(device)
        # lemmas = lemmas.to(device)
        x_len = x_len.to(device)
        postags = postags.to(device)
        adjm = adjm.to(device)
        y = y.to(device)

        y_, mask, ae_logits, ae_logits_key = model.forward(words, x_len, postags, entitylabels, adjm, entities,
                                                           label_i2s)
        loss_ed = model.calculate_loss_ed(y_, mask, y, weight)
        if len(ae_logits_key) > 0:
            loss_ae, predicted_events = model.calculate_loss_ae(ae_logits, ae_logits_key, events, x_len.size()[0])
            loss = loss_ed + hyps["loss_alpha"] * loss_ae
        else:
            loss = loss_ed
            predicted_events = [{} for _ in range(x_len.size()[0])]
        all_events_.extend(predicted_events)

        y__ = torch.max(y_, 2)[1].view(y.size()).tolist()
        y = y.tolist()

        add_tokens(words, y, y__, x_len, all_tokens, word_i2s, label_i2s)

        # unpad
        for i, ll in enumerate(x_len):
            y[i] = y[i][:ll]
            y__[i] = y__[i][:ll]
        bp, br, bf = tester.calculate_report(y, y__, transform=True)
        all_y.extend(y)
        all_y_.extend(y__)

        cnt += 1
        other_information = ""

        if need_backward:
            loss.backward()
            if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)

            optimizer.step()
            other_information = 'Iter[{}] loss: {:.6f} edP: {:.4f}% edR: {:.4f}% edF1: {:.4f}%'.format(cnt, loss.item(),
                                                                                                       bp * 100.0,
                                                                                                       br * 100.0,
                                                                                                       bf * 100.0)
        progressbar(cnt, MAX_STEP, other_information)
        running_loss += loss.item()

    if save_output:
        with open(save_output, "w", encoding="utf-8") as f:
            for tokens in all_tokens:
                for token in tokens:
                    # to match conll2000 format
                    f.write("%s %s %s\n" % (token.word, token.triggerLabel, token.predictedLabel))
                f.write("\n")

    running_loss = running_loss / cnt
    ep, er, ef = tester.calculate_report(all_y, all_y_, transform=False)
    ap, ar, af = tester.calculate_sets(all_events, all_events_)
    print()
    return running_loss, ep, er, ef, ap, ar, af
