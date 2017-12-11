import os
import sys
import argparse
import shutil
from utils import batch_utils
from utils import train_utils

import torch
import torch.nn as nn
import numpy as np

from data.datasets import UbuntuDataset, AndroidDataset
from models import CNN, LSTM
from data.embedding import Embedding

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--embedding', type=str, default='askubuntu')
parser.add_argument('--embed', type=int, default=200)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--android', action='store_true')

best_mrr = -1
best_auc = -1


def main():
    global args, best_mrr, best_auc
    args = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    print args

    corpus_file = 'data/askubuntu/text_tokenized.txt.gz'
    dataset = UbuntuDataset(corpus_file)
    corpus = dataset.get_corpus()

    if args.embedding == 'askubuntu':
        embedding_file = 'data/askubuntu/vector/vectors_pruned.200.txt.gz'
    else:
        embedding_file = 'data/glove/glove.pruned.txt.gz'

    embedding_iter = Embedding.iterator(embedding_file)
    embedding = Embedding(args.embed, embedding_iter)
    print 'Embeddings loaded.'

    corpus_ids = embedding.corpus_to_ids(corpus)
    padding_id = embedding.vocab_ids['<padding>']

    train_file = 'data/askubuntu/train_random.txt'
    train_data = dataset.read_annotations(train_file)

    dev_file = 'data/askubuntu/dev.txt'
    dev_data = dataset.read_annotations(dev_file, max_neg=-1)
    dev_batches = batch_utils.generate_eval_batches(
        corpus_ids, dev_data, padding_id)

    assert args.model in ['lstm', 'cnn']
    if args.model == 'lstm':
        model = LSTM(args.embed, args.hidden)
    else:
        model = CNN(args.embed, args.hidden)

    print model
    print 'Parameters: {}'.format(params(model))

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.MultiMarginLoss(margin=args.margin)

    if cuda_available:
        criterion = criterion.cuda()

    if args.load:
        if os.path.isfile(args.load):
            print 'Loading checkpoint.'
            checkpoint = torch.load(args.load)
            args.start_epoch = checkpoint['epoch']
            best_mrr = checkpoint.get('best_mrr', -1)
            best_auc = checkpoint.get('best_auc', -1)
            model.load_state_dict(checkpoint['state_dict'])

            print 'Loaded checkpoint at epoch {}.'.format(checkpoint['epoch'])
        else:
            print 'No checkpoint found here.'

    if args.eval:
        test_file = 'data/askubuntu/test.txt'
        test_data = dataset.read_annotations(test_file, max_neg=-1)
        test_batches = batch_utils.generate_eval_batches(
            corpus_ids, test_data, padding_id)

        print 'Evaluating on dev set.'
        train_utils.evaluate_metrics(
            args, model, embedding, dev_batches, padding_id)

        print 'Evaluating on test set.'
        train_utils.evaluate_metrics(
            args, model, embedding, test_batches, padding_id)
        return

    if args.android:
        android_file = 'data/android/corpus.tsv.gz'
        android_dataset = AndroidDataset(android_file)
        android_ids = embedding.corpus_to_ids(android_dataset.get_corpus())

        dev_pos_file = 'data/android/dev.pos.txt'
        dev_neg_file = 'data/android/dev.neg.txt'
        android_data = android_dataset.read_annotations(
            dev_pos_file, dev_neg_file)

        android_batches = batch_utils.generate_eval_batches(
            android_ids, android_data, padding_id)

    for epoch in xrange(args.start_epoch, args.epochs):
        train_batches = batch_utils.generate_train_batches(
            corpus_ids, train_data, args.batch_size, padding_id)

        train_utils.train(args, model, embedding, optimizer, criterion,
                          train_batches, padding_id, epoch)

        map, mrr, p1, p5 = train_utils.evaluate_metrics(
            args, model, embedding, dev_batches, padding_id)

        auc = -1
        if args.android:
            auc = train_utils.evaluate_auc(
                args, model, embedding, android_batches, padding_id)

        is_best = auc > best_auc if args.android else mrr > best_mrr
        best_mrr = max(mrr, best_mrr)
        best_auc = max(auc, best_auc)
        save(args, {
            'epoch': epoch + 1,
            'arch': 'lstm',
            'state_dict': model.state_dict(),
            'best_mrr': best_mrr,
            'best_auc': best_auc,
        }, is_best)


def params(model):
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in trainable])
    return params


def save(args, state, is_best):
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    latest = '{}.{}.{}.{}.latest.pth.tar'.format(
        args.model, args.hidden, int(args.margin * 100), args.embedding)
    latest = os.path.join(directory, latest)

    torch.save(state, latest)
    if is_best:
        best = '{}.{}.{}.{}.best.pth.tar'.format(
            args.model, args.hidden, int(args.margin * 100), args.embedding)
        best = os.path.join(directory, best)
        shutil.copyfile(latest, best)


if __name__ == '__main__':
    main()
