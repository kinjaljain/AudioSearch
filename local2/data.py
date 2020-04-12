import os
from io import open

class Corpus(object):
    def __init__(self, path):
        self.train = self.get_lines(os.path.join(path, 'train.txt'))
        self.valid = self.get_lines(os.path.join(path, 'valid.txt'))
        self.test = self.get_lines(os.path.join(path, 'test.txt'))

    def get_lines(self, path):
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
        output = []
        for line in lines:
            line = line.strip()
            if line.startswith('=') and line.endswith('='):
                continue
            l = line.split('.')[:-1]
            for i in l:
                i.strip()
            if len(l) < 5:
                continue
            output.extend(l)
        return output
