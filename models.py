import gzip
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

cuda_available = torch.cuda.is_available()


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        self.hidden = args.hidden
        self.embed = args.embed

        kernel = 3
        conv1d = nn.Conv1d(self.embed, self.hidden, kernel, padding=1)
        self.conv1d = conv1d.cuda() if cuda_available else conv1d

    def forward(self, input):
        # input = sequence x questions x embed
        assert self.embed == input.size(2)

        # input = questions x embed x sequence
        input = input.transpose(0, 2).transpose(0, 1)

        # output = questions x hidden x sequence
        output = self.conv1d(input)
        output = F.tanh(output)

        # output = sequence x questions x hidden
        output = output.transpose(0, 1).transpose(0, 2)

        return output


class LSTM(nn.Module):
    def __init__(self, args, bidirectional=False):
        super(LSTM, self).__init__()

        self.hidden = args.hidden
        self.embed = args.embed

        lstm = nn.LSTM(self.embed, self.hidden, bidirectional=bidirectional)
        self.lstm = lstm.cuda() if cuda_available else lstm

    def forward(self, input):
        # input = sequence x questions x embed
        seq_len, n_questions = input.size(0), input.size(1)
        assert self.embed == input.size(2)

        # h_c[0] = 1 x questions x hidden
        h_c = (Variable(torch.zeros(1, n_questions, self.hidden)),
               Variable(torch.zeros(1, n_questions, self.hidden)))

        # output = sequence x questions x hidden
        output = Variable(torch.zeros(seq_len, n_questions, self.hidden))

        if cuda_available:
            h_c = (h_c[0].cuda(), h_c[1].cuda())
            output = output.cuda()

        # output = sequence x questions x hidden
        for j in xrange(seq_len):
            _, h_c = self.lstm(input[j].view(1, n_questions, -1), h_c)
            output[j, :, :] = h_c[0]

        return output


class Embedding:
    def __init__(self, embed_size, iter, oov='<unk>', padding='<padding>'):
        vocab_ids = {oov: 0, padding: 1}
        words = [oov, padding]
        vectors = [np.zeros((embed_size, )),
                   np.random.uniform(-0.00005, 0.00005, (embed_size, ))]

        for word, vector in iter:
            vocab_ids[word] = len(vocab_ids)
            words.append(word)
            vectors.append(vector)

        # with gzip.open(file) as f:
        #     for line in f:
        #         if line.strip():
        #             word_vector = line.strip().split()
        #             word = word_vector[0]

        #             words.append(word)
        #             vectors.append(
        #                 np.array([float(x) for x in word_vector[1:]]))

        #             vocab_ids[word] = len(vocab_ids)

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
                    vectors = np.array([float(x) for x in word_vector[1:]])
                    yield word, vectors
