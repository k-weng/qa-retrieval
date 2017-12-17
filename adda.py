import os
import sys
import shutil
import argparse
from utils import batch_utils, train_utils

import torch
import torch.nn as nn

from data.datasets import UbuntuDataset, AndroidDataset
from models import LSTM, FFN, CNN
from data.embedding import Embedding

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('load', type=str)
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--embed', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--elr', type=float, default=1e-3)
parser.add_argument('--dlr', type=float, default=1e-3)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--batch_count', type=int, default=318)

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

    dev_pos_file = 'data/android/dev.pos.txt'
    dev_neg_file = 'data/android/dev.neg.txt'
    android_dev_data = android_dataset.read_annotations(
        dev_pos_file, dev_neg_file)

    android_dev_batches = batch_utils.generate_eval_batches(
        android_ids, android_dev_data, padding_id)

    assert args.model in ['lstm', 'cnn']
    if os.path.isfile(args.load):
        checkpoint = torch.load(args.load)
    else:
        print 'No checkpoint found here.'
        return

    if args.model == 'lstm':
        encoder_src = LSTM(embed_size, args.hidden)
        encoder_tgt = LSTM(embed_size, args.hidden)
    else:
        encoder_src = CNN(embed_size, args.hidden)
        encoder_tgt = CNN(embed_size, args.hidden)
    encoder_src.load_state_dict(checkpoint['state_dict'])
    encoder_src.eval()

    model_discrim = FFN(args.hidden)

    print encoder_src
    print encoder_tgt
    print model_discrim

    criterion = nn.CrossEntropyLoss()
    if cuda_available:
        criterion = criterion.cuda()

    betas = (0.5, 0.999)
    weight_decay = 1e-4
    optimizer_tgt = torch.optim.Adam(encoder_tgt.parameters(),
                                     lr=args.elr,
                                     betas=betas,
                                     weight_decay=weight_decay)
    optimizer_discrim = torch.optim.Adam(model_discrim.parameters(),
                                         lr=args.dlr,
                                         betas=betas,
                                         weight_decay=weight_decay)

    for epoch in xrange(args.start_epoch, args.epochs):
        train_batches = \
            batch_utils.generate_classifier_train_batches(
                ubuntu_ids, android_ids, args.batch_size,
                args.batch_count, padding_id)

        train_utils.train_adda(
            args, encoder_src, encoder_tgt, model_discrim, embedding,
            optimizer_tgt, optimizer_discrim, criterion,
            train_batches, padding_id, epoch)

        auc = train_utils.evaluate_auc(
            args, encoder_tgt, embedding, android_dev_batches, padding_id)

        is_best = auc > best_auc
        best_auc = max(auc, best_auc)
        save(args, {
            'epoch': epoch + 1,
            'arch': 'lstm',
            'encoder_tgt_state_dict': encoder_tgt.state_dict(),
            'discrim_state_dict': model_discrim.state_dict(),
            'best_auc': best_auc,
        }, is_best)


def save(args, state, is_best):
    directory = 'adda_models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    latest = '{}.{}.{}.latest.pth.tar'.format(
        args.model, args.hidden, int(args.margin * 100))
    latest = os.path.join(directory, latest)

    torch.save(state, latest)
    if is_best:
        best = '{}.{}.{}.best.pth.tar'.format(
            args.model, args.hidden, int(args.margin * 100))
        best = os.path.join(directory, best)
        shutil.copyfile(latest, best)


if __name__ == '__main__':
    main()
