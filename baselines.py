from gensim import summarization
from data.datasets import AndroidDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import torch.nn.functional as F
from utils.meter import AUCMeter


def bm25_auc(data, dataset):
    meter = AUCMeter()
    for batch in data:
        q = dataset.retrieve_combined(batch[0], joined=False)[0]
        p = dataset.retrieve_combined(batch[1], joined=False)

        bm25 = summarization.bm25.BM25(p)
        average_idf = sum(
            float(val) for val in bm25.idf.values()) / len(bm25.idf)

        scores = np.array(bm25.get_scores(q, average_idf))

        scores = torch.DoubleTensor(scores)
        target = torch.DoubleTensor(batch[2])

        meter.add(scores, target)

    return meter.value(0.05)


def tfidf_auc(data, dataset):
    meter = AUCMeter()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dataset.retrieve_combined())
    print dataset.retrieve_combined()

    for batch in data:
        q = vectorizer.transform(dataset.retrieve_combined(batch[0]))
        p = vectorizer.transform(dataset.retrieve_combined(batch[1]))

        q = torch.DoubleTensor(q.todense())
        p = torch.DoubleTensor(p.todense())

        print q
        print p
        scores = F.cosine_similarity(q, p, dim=1).cpu()
        target = torch.DoubleTensor(batch[2])

        meter.add(scores, target)
        break

    return meter.value(0.05)


def main():
    corpus_file = 'data/android/corpus.tsv.gz'
    dataset = AndroidDataset(corpus_file)

    dev_pos_file = 'data/android/dev.pos.txt'
    dev_neg_file = 'data/android/dev.neg.txt'
    dev_data = dataset.read_annotations(dev_pos_file, dev_neg_file)

    test_pos_file = 'data/android/test.pos.txt'
    test_neg_file = 'data/android/test.neg.txt'
    test_data = dataset.read_annotations(test_pos_file, test_neg_file)

    dev_tfidf_score = tfidf_auc(dev_data, dataset)
    print 'TFIDF: {}'.format(dev_tfidf_score)

    test_tfidf_score = tfidf_auc(test_data, dataset)
    print 'TFIDF: {}'.format(test_tfidf_score)


if __name__ == '__main__':
    main()
