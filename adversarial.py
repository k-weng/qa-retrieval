import os
import sys
import argparse
from utils import batch_utils, train_utils

import torch
import torch.nn as nn
import numpy as np

from data.datasets import UbuntuDataset, AndroidDataset
from models import LSTM, FFN
from data.embedding import Embedding

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--embed', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--elr', type=float, default=0.001)
parser.add_argument('--clr', type=float, default=-0.001)
parser.add_argument('--llambda', type=float, default=0.5)


def main():
    global args, best_mrr, best_auc
    args = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    print args

    embedding_file = 'data/glove/glove.pruned.txt.gz'
    embedding_iter = Embedding.iterator(embedding_file)
    embed_size = 300
    embedding = Embedding(embed_size, embedding_iter)
    print 'Embeddings loaded.'

    android_corpus_file = 'data/android/corpus.tsv.gz'
    android_dataset = AndroidDataset(android_corpus_file)
    android_corpus = android_dataset.get_corpus()
    android_ids = embedding.corpus_to_ids(android_corpus)
    print 'Got Android corpus ids.'

    ubuntu_corpus_file = 'data/askubuntu/text_tokenized.txt.gz'
    ubuntu_dataset = UbuntuDataset(ubuntu_corpus_file)
    ubuntu_corpus = ubuntu_dataset.get_corpus()
    ubuntu_ids = embedding.corpus_to_ids(ubuntu_corpus)
    print 'Got AskUbuntu corpus ids.'

    padding_id = embedding.vocab_ids['<padding>']

    ubuntu_train_file = 'data/askubuntu/train_random.txt'
    ubuntu_train_data = ubuntu_dataset.read_annotations(ubuntu_train_file)

    dev_pos_file = 'data/android/dev.pos.txt'
    dev_neg_file = 'data/android/dev.neg.txt'
    android_data = android_dataset.read_annotations(
        dev_pos_file, dev_neg_file)

    android_batches = batch_utils.generate_eval_batches(
        android_ids, android_data, padding_id)

    model_encoder = LSTM(embed_size, args.hidden)
    model_classifier = FFN(args.hidden)
    print model_encoder
    print model_classifier

    optimizer_encoder = torch.optim.Adam(
        model_encoder.parameters(), args.elr)
    criterion_encoder = nn.MultiMarginLoss(margin=args.margin)

    optimizer_classifier = torch.optim.Adam(
        model_classifier.parameters(), args.clr)
    criterion_classifier = nn.BCELoss()

    if cuda_available:
        criterion_encoder = criterion_encoder.cuda()
        criterion_classifier = criterion_classifier.cuda()

    for epoch in xrange(args.start_epoch, args.epochs):
        encoder_train_batches = batch_utils.generate_train_batches(
            ubuntu_ids, ubuntu_train_data,
            args.batch_size, padding_id)
        classifier_train_batches = \
            batch_utils.generate_classifier_train_batches(
                ubuntu_ids, android_ids, args.batch_size,
                len(encoder_train_batches), padding_id)

        train_utils.train_encoder_classifer(
            args, model_encoder, model_classifier, embedding,
            optimizer_encoder, optimizer_classifier,
            criterion_encoder, criterion_classifier,
            zip(encoder_train_batches, classifier_train_batches),
            padding_id, epoch, args.llambda)

        train_utils.evaluate_auc(
            args, model_encoder, embedding, android_batches, padding_id)
        break


if __name__ == '__main__':
    main()
