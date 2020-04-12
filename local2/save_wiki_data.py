from data import Corpus
import os

path = "wikitext-2"
corpus = Corpus(path)
f = open(os.path.join(path, 'train_.txt'), 'w')
f.writelines(("%s\n" % t for t in corpus.train))
f = open(os.path.join(path, 'test_.txt'), 'w')
f.writelines(("%s\n" % t for t in corpus.test))
f = open(os.path.join(path, 'valid_.txt'), 'w')
f.writelines(("%s\n" % t for t in corpus.valid))

