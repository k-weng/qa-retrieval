import gzip
import torch
import numpy as np


class Embedding:
    def __init__(self, embed_size, file, oov='<unk>', padding='<padding>'):
        vocab_ids = {oov: 0, padding: 1}
        words = [oov, padding]
        vectors = [np.zeros((embed_size, )),
                   np.random.uniform(-0.00005, 0.00005, (embed_size, ))]

        with gzip.open(file) as f:
            for line in f:
                if line.strip():
                    word_vector = line.strip().split()
                    word = word_vector[0]

                    words.append(word)
                    vectors.append(
                        np.array([float(x) for x in word_vector[1:]]))

                    vocab_ids[word] = len(vocab_ids)

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
