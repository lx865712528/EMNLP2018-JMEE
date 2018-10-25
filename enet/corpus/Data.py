import json
from collections import Counter, OrderedDict

import six
import torch
from torchtext.data import Field, Example, Pipeline, Dataset

from enet.corpus.Corpus import Corpus
from enet.corpus.Sentence import Sentence


class SparseField(Field):
    def process(self, batch, device, train):
        return batch


class EntityField(Field):
    '''
    Processing data each sentence has only one

    [(2, 3, "entity_type")]
    '''

    def preprocess(self, x):
        return x

    def pad(self, minibatch):
        return minibatch

    def numericalize(self, arr, device=None, train=True):
        return arr


class EventField(Field):
    '''
    Processing data each sentence has only one

    {
            (2, 3, "event_type_str") --> [(1, 2, "role_type_str"), ...]
            ...
    }
    '''

    def preprocess(self, x):
        return x

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                for key, value in x.items():
                    for v in value:
                        counter.update([v[2]])
        self.vocab = self.vocab_cls(counter, specials=["OTHER"], **kwargs)

    def pad(self, minibatch):
        return minibatch

    def numericalize(self, arr, device=None, train=True):
        if self.use_vocab:
            arr = [{key: [(v[0], v[1], self.vocab.stoi[v[2]]) for v in value] for key, value in dd.items()} for dd in
                   arr]
        return arr


class MultiTokenField(Field):
    '''
    Processing data like "[ ["A", "A", "A"], ["A", "A"], ["A", "A"], ["A"] ]"
    '''

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and
                not isinstance(x, six.text_type)):  # never
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):  # never
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = [Pipeline(six.text_type.lower)(xx) for xx in x]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                for xx in x:
                    counter.update(xx)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [[self.pad_token]] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [[self.init_token]]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [[self.eos_token]]))
            else:
                padded.append(
                    ([] if self.init_token is None else [[self.init_token]]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [[self.eos_token]]) +
                    [[self.pad_token]] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr, device=None, train=True):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[[self.vocab.stoi[xx] for xx in x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)

        if self.include_lengths:
            return arr, lengths
        return arr


class ACE2005Dataset(Corpus):
    """
    Defines a dataset composed of Examples along with its Fields.
    """

    sort_key = None

    def __init__(self, path, fields, keep_events=None, only_keep=False, **kwargs):
        '''
        Create a corpus given a path, field list, and a filter function.

        :param path: str, Path to the data file
        :param fields: dict[str: tuple(str, Field)],
                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
        :param keep_events: int, minimum sentence events. Default keep all.
        '''
        self.keep_events = keep_events
        self.only_keep = only_keep
        super(ACE2005Dataset, self).__init__(path, fields, **kwargs)

    def parse_example(self, path, fields):
        examples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                jl = json.loads(line, encoding="utf-8")
                for js in jl:
                    ex = self.parse_sentence(js, fields)
                    if ex is not None:
                        examples.append(ex)

        return examples

    def parse_sentence(self, js, fields):
        WORDS = fields["words"]
        POSTAGS = fields["pos-tags"]
        # LEMMAS = fields["lemma"]
        ENTITYLABELS = fields["golden-entity-mentions"]
        ADJMATRIX = fields["stanford-colcc"]
        LABELS = fields["golden-event-mentions"]
        EVENTS = fields["all-events"]
        ENTITIES = fields["all-entities"]

        sentence = Sentence(json_content=js)
        ex = Example()
        setattr(ex, WORDS[0], WORDS[1].preprocess(sentence.wordList))
        setattr(ex, POSTAGS[0], POSTAGS[1].preprocess(sentence.posLabelList))
        # setattr(ex, LEMMAS[0], LEMMAS[1].preprocess(sentence.lemmaList))
        setattr(ex, ENTITYLABELS[0], ENTITYLABELS[1].preprocess(sentence.entityLabelList))
        setattr(ex, ADJMATRIX[0], (sentence.adjpos, sentence.adjv))
        setattr(ex, LABELS[0], LABELS[1].preprocess(sentence.triggerLabelList))
        setattr(ex, EVENTS[0], EVENTS[1].preprocess(sentence.events))
        setattr(ex, ENTITIES[0], ENTITIES[1].preprocess(sentence.entities))

        if self.keep_events is not None:
            if self.only_keep and sentence.containsEvents != self.keep_events:
                return None
            elif not self.only_keep and sentence.containsEvents < self.keep_events:
                return None
            else:
                return ex
        else:
            return ex

    def longest(self):
        return max([len(x.POSTAGS) for x in self.examples])
