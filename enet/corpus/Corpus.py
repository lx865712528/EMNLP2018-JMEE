from torchtext.data import Dataset


class Corpus(Dataset):
    def __init__(self, path, fields, **kwargs):
        '''
        Create a corpus given a path, field list, and a filter function.

        :param path: str, Path to the data file
        :param fields: dict[str: tuple(str, Field)],
                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
        '''
        self.path = path
        self._size = None

        examples = self.parse_example(path, fields)
        fields = list(fields.values())
        super(Corpus, self).__init__(examples, fields, **kwargs)

    def parse_example(self, path, fields):
        raise NotImplementedError