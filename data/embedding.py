import gzip
import numpy as np

import torch


class Embedding:
    def __init__(self, embed_size, iter, oov='<unk>', padding='<padding>'):
        vocab_ids = {oov: 0, padding: 1}
        words = [oov, padding]
        vectors = [np.zeros((embed_size, )), np.zeros((embed_size, ))]

        for word, vector in iter:
            vocab_ids[word] = len(vocab_ids)
            words.append(word)
            vectors.append(np.array([float(x) for x in vector]))

        self.vocab_ids = vocab_ids
        self.oov_id = vocab_ids[oov]
        self.words = words
        self.embed_size = embed_size
        self.embeddings = np.array(vectors)

    def words_to_ids(self, words):
        return np.array(
            filter(lambda x: x != self.oov_id,
                   [self.vocab_ids.get(x, self.oov_id) for x in words]))

    def ids_to_words(self, ids):
        n_words, words = len(self.vocab_ids), self.words
        return [words[i] if i < n_words else "<err>" for i in ids]

    def corpus_to_ids(self, corpus, max_len=100):
        corpus_ids = {}
        for id, (title, body) in corpus.iteritems():
            corpus_ids[id] = (self.words_to_ids(title),
                              self.words_to_ids(body)[:max_len])
        return corpus_ids

    def get_embeddings(self, ids):
        return self.embeddings[torch.LongTensor(ids)]

    @staticmethod
    def iterator(file):
        fopen = gzip.open if file.endswith(".gz") else open
        with fopen(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    word_vector = line.split()
                    word = word_vector[0]
                    vectors = word_vector[1:]
                    yield word, vectors
