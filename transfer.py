import os
import sys
import argparse

from utils import batch_utils
from utils import train_utils

import torch

from datasets import AndroidDataset
from models import CNN, LSTM, Embedding


parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('load', type=str)
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--embed', type=int, default=200)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--margin', type=float, default=0.2)


def main():
    args = parser.parse_args()
    print args

    corpus_file = 'android/corpus.tsv.gz'
    dataset = AndroidDataset(corpus_file)
    corpus = dataset.get_corpus()

    embedding_file = 'askubuntu/vector/vectors_pruned.200.txt.gz'
    embedding_iter = Embedding.iterator(embedding_file)
    embedding = Embedding(args.embed, embedding_iter)
    print 'Embeddings loaded.'

    corpus_ids = embedding.corpus_to_ids(corpus)
    padding_id = embedding.vocab_ids['<padding>']

    dev_pos_file = 'android/dev.pos.txt'
    dev_neg_file = 'android/dev.neg.txt'
    dev_data = dataset.read_annotations(dev_pos_file, dev_neg_file)

    test_pos_file = 'android/test.pos.txt'
    test_neg_file = 'android/test.neg.txt'
    test_data = dataset.read_annotations(test_pos_file, test_neg_file)

    dev_batches = batch_utils.generate_eval_batches(
        corpus_ids, dev_data, padding_id)
    test_batches = batch_utils.generate_eval_batches(
        corpus_ids, test_data, padding_id)

    if os.path.isfile(args.load):
        if args.model == 'lstm':
            model = LSTM(args)
        else:
            model = CNN(args)

        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print 'No checkpoint found here.'

    print 'Evaluating on dev set.'
    train_utils.evaluate_auc(
        args, model, embedding, dev_batches, padding_id)

    print 'Evaluating on test set.'
    train_utils.evaluate_auc(
        args, model, embedding, test_batches, padding_id)
    return


if __name__ == '__main__':
    main()
