import sys
import gzip
from datasets import AndroidDataset, UbuntuDataset
from embedding import Embedding


def prune(file):
    ubuntu_dataset = UbuntuDataset('askubuntu/text_tokenized.txt.gz')
    android_dataset = AndroidDataset('android/corpus.tsv.gz')

    f = gzip.open('{}.pruned.txt.gz'.format(file.split('.')[0]), 'w')
    embeddings = Embedding.iterator(file)

    words = set()
    for title, body in ubuntu_dataset.corpus.values():
        words.update(title)
        words.update(body)
    for title, body in android_dataset.corpus.values():
        words.update(title)
        words.update(body)

    for word, vector in embeddings:
        if word in words:
            f.write(' '.join([word] + vector))
            f.write('\n')

    f.close()


if __name__ == '__main__':
    file = sys.argv[1]
    print file
    # prune(file)
