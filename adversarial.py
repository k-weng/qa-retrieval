import os
import sys
import shutil
import argparse
from utils import batch_utils, train_utils

import torch
import torch.nn as nn

from data.datasets import UbuntuDataset, AndroidDataset
from models import LSTM, FFN
from data.embedding import Embedding

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--embed', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--elr', type=float, default=0.001)
parser.add_argument('--clr', type=float, default=-0.001)
parser.add_argument('--lmbda', type=float, default=1e-4)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--eval', action='store_true')

best_auc = -1


def main():
    global args, best_auc
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
    android_dev_data = android_dataset.read_annotations(
        dev_pos_file, dev_neg_file)

    android_dev_batches = batch_utils.generate_eval_batches(
        android_ids, android_dev_data, padding_id)

    model_encoder = LSTM(embed_size, args.hidden)
    model_classifier = FFN(args.hidden)
    print model_encoder
    print model_classifier

    optimizer_encoder = torch.optim.Adam(
        model_encoder.parameters(), lr=args.elr)
    criterion_encoder = nn.MultiMarginLoss(margin=args.margin)

    optimizer_classifier = torch.optim.Adam(
        model_classifier.parameters(), lr=args.clr)
    criterion_classifier = nn.CrossEntropyLoss()

    if cuda_available:
        criterion_encoder = criterion_encoder.cuda()
        criterion_classifier = criterion_classifier.cuda()

    if args.load:
        if os.path.isfile(args.load):
            print 'Loading checkpoint.'
            checkpoint = torch.load(args.load)
            args.start_epoch = checkpoint['epoch']
            best_auc = checkpoint.get('best_auc', -1)
            model_encoder.load_state_dict(
                checkpoint['encoder_state_dict'])
            model_classifier.load_state_dict(
                checkpoint['classifier_state_dict'])

            print 'Loaded checkpoint at epoch {}.'.format(checkpoint['epoch'])
        else:
            print 'No checkpoint found here.'

    if args.eval:
        test_pos_file = 'data/android/test.pos.txt'
        test_neg_file = 'data/android/test.neg.txt'
        android_test_data = android_dataset.read_annotations(
            test_pos_file, test_neg_file)

        android_test_batches = batch_utils.generate_eval_batches(
            android_ids, android_test_data, padding_id)

        print 'Evaluating on dev set.'
        train_utils.evaluate_metrics(
            args, model_encoder, embedding,
            android_dev_batches, padding_id)

        print 'Evaluating on test set.'
        train_utils.evaluate_metrics(
            args, model_encoder, embedding,
            android_test_batches, padding_id)
        return

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
            padding_id, epoch, args.lmbda)

        auc = train_utils.evaluate_auc(
            args, model_encoder, embedding, android_dev_batches, padding_id)

        is_best = auc > best_auc
        best_auc = max(auc, best_auc)
        save(args, {
            'epoch': epoch + 1,
            'arch': 'lstm',
            'encoder_state_dict': model_encoder.state_dict(),
            'classifier_state_dict': model_classifier.state_dict(),
            'best_auc': best_auc,
        }, is_best)


def save(args, state, is_best):
    directory = 'adversarial_models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    latest = '{}.{}.{}.{}.latest.pth.tar'.format(
        args.model, args.hidden, int(args.margin * 100), args.lmbda)
    latest = os.path.join(directory, latest)

    torch.save(state, latest)
    if is_best:
        best = '{}.{}.{}.{}.best.pth.tar'.format(
            args.model, args.hidden, int(args.margin * 100), args.lmbda)
        best = os.path.join(directory, best)
        shutil.copyfile(latest, best)


if __name__ == '__main__':
    main()
