import os
import sys
import argparse

from utils import batch_utils
from utils import train_utils

import torch

from data.datasets import AndroidDataset
from models import CNN, LSTM
from data.embedding import Embedding

def main():

    corpus_file = 'data/android/corpus.tsv.gz'
    dataset = AndroidDataset(corpus_file)
    corpus = dataset.get_corpus()

    embedding_file = 'data/glove/glove.pruned.txt.gz'

    embedding_iter = Embedding.iterator(embedding_file)
    embedding = Embedding(args.embed, embedding_iter)
    print 'Embeddings loaded.'

    corpus_ids = embedding.corpus_to_ids(corpus)
    padding_id = embedding.vocab_ids['<padding>']

    dev_pos_file = 'data/android/dev.pos.txt'
    dev_neg_file = 'data/android/dev.neg.txt'
    dev_data = dataset.read_annotations(dev_pos_file, dev_neg_file)

    test_pos_file = 'data/android/test.pos.txt'
    test_neg_file = 'data/android/test.neg.txt'
    test_data = dataset.read_annotations(test_pos_file, test_neg_file)

    dev_batches = batch_utils.generate_eval_batches(
        corpus_ids, dev_data, padding_id)
    test_batches = batch_utils.generate_eval_batches(
        corpus_ids, test_data, padding_id)

main()
