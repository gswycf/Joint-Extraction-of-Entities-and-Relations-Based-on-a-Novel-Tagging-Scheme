import string
import pickle


class Index(object):
    def __init__(self):
        self.key2idx = {}
        self.idx2key = []

    def add(self, key):
        if key not in self.key2idx:
            self.key2idx[key] = len(self.idx2key)
            self.idx2key.append(key)
        return self.key2idx[key]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.key2idx[key]
        if isinstance(key, int):
            return self.idx2key[key]

    def __len__(self):
        return len(self.idx2key)

    def save(self, f):
        with open(f, 'wt', encoding='utf-8') as fout:
            for index, key in enumerate(self.idx2key):
                fout.write(key + '\t' + str(index) + '\n')

    def load(self, f):
        with open(f, 'rt', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                key = line.split()[0]
                self.add(key)


class Charset(Index):
    def __init__(self):
        super().__init__()
        for char in string.printable[0:-6]:#所有的字母加符号
            self.add(char)
        self.add("<pad>")
        self.add("<unk>")

    @staticmethod
    def type(char):
        if char in string.digits:
            return "Digits"
        if char in string.ascii_lowercase:
            return "Lower Case"
        if char in string.ascii_uppercase:
            return "Upper Case"
        if char in string.punctuation:
            return "Punctuation"
        return "Other"

    def __getitem__(self, key):
        if isinstance(key, str) and key not in self.key2idx:
            return self.key2idx["<unk>"]
        return super().__getitem__(key)


class Vocabulary(Index):
    def __init__(self):
        super().__init__()
        self.add("<pad>")
        self.add("<unk>")

    def __getitem__(self, key):
        if isinstance(key, str) and key not in self.key2idx:
            return self.key2idx["<unk>"]
        return super().__getitem__(key)


def prepare_sequence(seq, to_idx):
    return [to_idx[key] for key in seq]


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def time_display(s):
    d = s // (3600*24)
    s -= d * (3600*24)
    h = s // 3600
    s -= h * 3600
    m = s // 60
    s -= m * 60
    str_time = "{:1d}d ".format(int(d)) if d else "   "
    return str_time + "{:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s))