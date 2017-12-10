import os
import sys
import argparse
import shutil
from utils import batch_utils
from utils import train_utils

import torch
import torch.nn as nn
import numpy as np

from data.datasets import UbuntuDataset
from models import CNN, LSTM
from data.embedding import Embedding

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--embed_file', type=str,
                    default='data/askubuntu/vector/vectors_pruned.200.txt.gz')
parser.add_argument('--embed', type=int, default=200)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--eval', action='store_true')

best_mrr = -1


def main():
    global args, best_mrr
    args = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    print args

    corpus_file = 'data/askubuntu/text_tokenized.txt.gz'
    dataset = UbuntuDataset(corpus_file)
    corpus = dataset.get_corpus()

    embedding_iter = Embedding.iterator(args.embed_file)
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
        model = LSTM(args)
    else:
        model = CNN(args)

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
            best_mrr = checkpoint['best_mrr']
            model.load_state_dict(checkpoint['state_dict'])

            print 'Loaded checkpoint at epoch {}.'.format(checkpoint['epoch'])
        else:
            print 'No checkpoint found here.'

    if args.eval:
        test_file = 'askubuntu/test.txt'
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

    for epoch in xrange(args.start_epoch, args.epochs):
        train_batches = batch_utils.generate_train_batches(
            corpus_ids, train_data, args.batch_size, padding_id)

        train_utils.train(args, model, embedding, optimizer, criterion,
                          train_batches, padding_id, epoch)

        map, mrr, p1, p5 = train_utils.evaluate_metrics(
            args, model, embedding, dev_batches, padding_id)

        is_best = mrr > best_mrr
        best_mrr = max(mrr, best_mrr)
        save(args, {
            'epoch': epoch + 1,
            'arch': 'lstm',
            'state_dict': model.state_dict(),
            'best_mrr': best_mrr,
        }, is_best)


def params(model):
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in trainable])
    return params


def save(args, state, is_best):
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    latest = '{}_{}_{}_latest.pth.tar'.format(
        args.model, args.hidden, int(args.margin * 100))
    latest = os.path.join(directory, latest)

    torch.save(state, latest)
    if is_best:
        best = '{}_{}_{}_best.pth.tar'.format(
            args.model, args.hidden, int(args.margin * 100))
        best = os.path.join(directory, best)
        shutil.copyfile(latest, best)


if __name__ == '__main__':
    main()
