import argparse
import os
import pickle
import sys
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchtext.data import Field
from torchtext.vocab import Vectors

from enet import consts
from enet.corpus.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from enet.models.ee import EDModel
from enet.testing import EDTester
from enet.training import train
from enet.util import log


class EERunner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test", help="validation set")
        parser.add_argument("--train", help="training set", required=False)
        parser.add_argument("--dev", help="development set", required=False)
        parser.add_argument("--webd", help="word embedding", required=False)

        parser.add_argument("--batch", help="batch size", default=128, type=int)
        parser.add_argument("--epochs", help="n of epochs", default=sys.maxsize, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--optimizer", default="adam")
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--l2decay", default=0, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--finetune", help="pretrained model path")
        parser.add_argument("--earlystop", default=999999, type=int)
        parser.add_argument("--restart", default=999999, type=int)

        parser.add_argument("--device", default="cpu")
        parser.add_argument("--hps", help="model hyperparams", required=False)

        self.a = parser.parse_args()

    def set_device(self, device="cpu"):
        self.device = torch.device(device)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def load_model(self, fine_tune):

        if fine_tune is None:
            return EDModel(self.a.hps, self.get_device())
        else:
            mymodel = EDModel(self.a.hps)
            mymodel.load_model(fine_tune)
            mymodel.to(self.get_device())
            return mymodel

    def get_tester(self, voc_i2s):
        return EDTester(voc_i2s)

    def run(self):
        print("Running on", self.a.device)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)

        # create training set
        if self.a.train:
            log('loading corpus from %s' % self.a.train)

        WordsField = Field(lower=True, include_lengths=True, batch_first=True)
        PosTagsField = Field(lower=True, batch_first=True)
        EntityLabelsField = MultiTokenField(lower=False, batch_first=True)
        AdjMatrixField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        LabelField = Field(lower=False, batch_first=True, pad_token=None, unk_token=None)
        EventsField = EventField(lower=False, batch_first=True)
        EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)

        train_set = ACE2005Dataset(path=self.a.train,
                                   fields={"words": ("WORDS", WordsField),
                                           "pos-tags": ("POSTAGS", PosTagsField),
                                           "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                           "stanford-colcc": ("ADJM", AdjMatrixField),
                                           "golden-event-mentions": ("LABEL", LabelField),
                                           "all-events": ("EVENT", EventsField),
                                           "all-entities": ("ENTITIES", EntitiesField)},
                                   keep_events=1)

        dev_set = ACE2005Dataset(path=self.a.dev,
                                 fields={"words": ("WORDS", WordsField),
                                         "pos-tags": ("POSTAGS", PosTagsField),
                                         "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                         "stanford-colcc": ("ADJM", AdjMatrixField),
                                         "golden-event-mentions": ("LABEL", LabelField),
                                         "all-events": ("EVENT", EventsField),
                                         "all-entities": ("ENTITIES", EntitiesField)},
                                 keep_events=0)

        test_set = ACE2005Dataset(path=self.a.test,
                                  fields={"words": ("WORDS", WordsField),
                                          "pos-tags": ("POSTAGS", PosTagsField),
                                          "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                          "stanford-colcc": ("ADJM", AdjMatrixField),
                                          "golden-event-mentions": ("LABEL", LabelField),
                                          "all-events": ("EVENT", EventsField),
                                          "all-entities": ("ENTITIES", EntitiesField)},
                                  keep_events=0)

        if self.a.webd:
            pretrained_embedding = Vectors(self.a.webd, ".", unk_init=partial(torch.nn.init.uniform_, a=-0.15, b=0.15))
            WordsField.build_vocab(train_set.WORDS, dev_set.WORDS, vectors=pretrained_embedding)
        else:
            WordsField.build_vocab(train_set.WORDS, dev_set.WORDS)
        PosTagsField.build_vocab(train_set.POSTAGS, dev_set.POSTAGS)
        EntityLabelsField.build_vocab(train_set.ENTITYLABELS, dev_set.ENTITYLABELS)
        LabelField.build_vocab(train_set.LABEL, dev_set.LABEL)
        EventsField.build_vocab(train_set.EVENT, dev_set.EVENT)

        consts.O_LABEL = LabelField.vocab.stoi["O"]
        # print("O label is", consts.O_LABEL)
        consts.ROLE_O_LABEL = EventsField.vocab.stoi["OTHER"]
        # print("O label for AE is", consts.ROLE_O_LABEL)

        dev_set1 = ACE2005Dataset(path=self.a.dev,
                                  fields={"words": ("WORDS", WordsField),
                                          "pos-tags": ("POSTAGS", PosTagsField),
                                          "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                          "stanford-colcc": ("ADJM", AdjMatrixField),
                                          "golden-event-mentions": ("LABEL", LabelField),
                                          "all-events": ("EVENT", EventsField),
                                          "all-entities": ("ENTITIES", EntitiesField)},
                                  keep_events=1, only_keep=True)

        test_set1 = ACE2005Dataset(path=self.a.test,
                                   fields={"words": ("WORDS", WordsField),
                                           "pos-tags": ("POSTAGS", PosTagsField),
                                           "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                           "stanford-colcc": ("ADJM", AdjMatrixField),
                                           "golden-event-mentions": ("LABEL", LabelField),
                                           "all-events": ("EVENT", EventsField),
                                           "all-entities": ("ENTITIES", EntitiesField)},
                                   keep_events=1, only_keep=True)

        print("dev set length", len(dev_set))
        print("dev set 1/1 length", len(dev_set1))

        print("test set length", len(test_set))
        print("test set 1/1 length", len(test_set1))

        self.a.label_weight = torch.ones([len(LabelField.vocab.itos)]) * 5
        self.a.label_weight[consts.O_LABEL] = 1.0

        self.a.hps = eval(self.a.hps)
        if "wemb_size" not in self.a.hps:
            self.a.hps["wemb_size"] = len(WordsField.vocab.itos)
        if "pemb_size" not in self.a.hps:
            self.a.hps["pemb_size"] = len(PosTagsField.vocab.itos)
        if "psemb_size" not in self.a.hps:
            self.a.hps["psemb_size"] = max([train_set.longest(), dev_set.longest(), test_set.longest()]) + 2
        if "eemb_size" not in self.a.hps:
            self.a.hps["eemb_size"] = len(EntityLabelsField.vocab.itos)
        if "oc" not in self.a.hps:
            self.a.hps["oc"] = len(LabelField.vocab.itos)
        if "ae_oc" not in self.a.hps:
            self.a.hps["ae_oc"] = len(EventsField.vocab.itos)

        tester = self.get_tester(LabelField.vocab.itos)

        if self.a.finetune:
            log('init model from ' + self.a.finetune)
            model = self.load_model(self.a.finetune)
            log('model loaded, there are %i sets of params' % len(model.parameters_requires_grads()))
        else:
            model = self.load_model(None)
            log('model created from scratch, there are %i sets of params' % len(model.parameters_requires_grads()))

        if self.a.optimizer == "adadelta":
            optimizer_constructor = partial(torch.optim.Adadelta, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        elif self.a.optimizer == "adam":
            optimizer_constructor = partial(torch.optim.Adam, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        else:
            optimizer_constructor = partial(torch.optim.SGD, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay,
                                            momentum=0.9)

        log('optimizer in use: %s' % str(self.a.optimizer))

        if not os.path.exists(self.a.out):
            os.mkdir(self.a.out)
        with open(os.path.join(self.a.out, "word.vec"), "wb") as f:
            pickle.dump(WordsField.vocab, f)
        with open(os.path.join(self.a.out, "pos.vec"), "wb") as f:
            pickle.dump(PosTagsField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "entity.vec"), "wb") as f:
            pickle.dump(EntityLabelsField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "label.vec"), "wb") as f:
            pickle.dump(LabelField.vocab.stoi, f)
        with open(os.path.join(self.a.out, "role.vec"), "wb") as f:
            pickle.dump(EventsField.vocab.stoi, f)

        log('init complete\n')

        self.a.word_i2s = WordsField.vocab.itos
        self.a.label_i2s = LabelField.vocab.itos
        self.a.role_i2s = EventsField.vocab.itos
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        train(
            model=model,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set,
            optimizer_constructor=optimizer_constructor,
            epochs=self.a.epochs,
            tester=tester,
            parser=self.a,
            other_testsets={
                "dev 1/1": dev_set1,
                "test 1/1": test_set1,
            }
        )
        log('Done!')


if __name__ == "__main__":
    EERunner().run()
