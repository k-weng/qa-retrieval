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
embedding = Embedding(args.embed, embedding_iter)
print 'Embeddings loaded.'

android_corpus_file = 'data/android/corpus.tsv.gz'
android_dataset = AndroidDataset(android_corpus_file)
android_corpus = android_dataset.get_corpus()

ubuntu_corpus_file = 'data/askubuntu/text_tokenized.txt.gz'
ubuntu_dataset = UbuntuDataset(ubuntu_corpus_file)
ubuntu_corpus = ubuntu_dataset.get_corpus()

batch_size = 20
batch_count = 10
batches2, batches2_labels = batch_utils.generate_adv_domain_train_batches(ubuntu_corpus,
                                                                        android_corpus, 
                                                                        batch_size, 
                                                                        batch_count)
