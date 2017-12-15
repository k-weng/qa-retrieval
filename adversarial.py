import os
import sys
import argparse

from utils import batch_utils
from utils import train_utils

import torch

from data.datasets import UbuntuDataset, AndroidDataset
from models import CNN, LSTM
from data.embedding import Embedding

embedding_file = 'data/glove/glove.pruned.txt.gz'
embedding_iter = Embedding.iterator(embedding_file)
embed_size = 300
embedding = Embedding(embed_size, embedding_iter)
print 'Embeddings loaded.'

android_corpus_file = 'data/android/corpus.tsv.gz'
android_dataset = AndroidDataset(android_corpus_file)
android_corpus = android_dataset.get_corpus()
android_corpus_ids = embedding.corpus_to_ids(android_corpus)
print 'Android Corpus ids'

ubuntu_corpus_file = 'data/askubuntu/text_tokenized.txt.gz'
ubuntu_dataset = UbuntuDataset(ubuntu_corpus_file)
ubuntu_corpus = ubuntu_dataset.get_corpus()
ubuntu_corpus_ids = embedding.corpus_to_ids(android_corpus)
print 'Ubuntu Corpus ids'

padding_id = embedding.vocab_ids['<padding>']

batch_size = 20
batch_count = 10
#batches 2 is list of batch
#each batch is (titles, bodies, labels)
#titles is n x (2*batch_size), where n is padded sequence length
batches2 = batch_utils.generate_adv_domain_train_batches(ubuntu_corpus_ids, android_corpus_ids, batch_size, batch_count, padding_id)
