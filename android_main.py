from gensim import summarization
from datasets import AndroidDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import torch.nn.functional as F
from meter import AUCMeter


def bm25_auc(data, dataset):
    meter = AUCMeter()
    for batch in data:
        query_t, query_b = dataset.get_sequence(batch[0])
        titles, bodies = dataset.get_sequences(batch[1])

        sequences = [title + body for title, body in zip(titles, bodies)]

        bm25 = summarization.bm25.BM25(sequences)
        average_idf = sum(
            float(val) for val in bm25.idf.values()) / len(bm25.idf)

        scores = np.array(bm25.get_scores(query_t + query_b, average_idf))

        scores = torch.DoubleTensor(scores)
        target = torch.DoubleTensor(batch[2])

        meter.add(scores, target)

    return meter.value(0.5)


def tfidf_auc(data, dataset):
    meter = AUCMeter()
    vectorizer = TfidfVectorizer()

    for batch in data:
        sequences = dataset.get_joined_sequences(
            [batch[0]] + batch[1])

        tfidf_weighted = vectorizer.fit_transform(sequences)
        tfidf_weighted = torch.DoubleTensor(tfidf_weighted.todense())

        q = tfidf_weighted[0].unsqueeze(0)
        p = tfidf_weighted[1:]

        scores = F.cosine_similarity(q, p, dim=1).cpu()
        target = torch.DoubleTensor(batch[2])

        meter.add(scores, target)

    return meter.value(0.5)


def main():
    corpus_file = 'android/corpus.tsv.gz'
    dataset = AndroidDataset(corpus_file)

    dev_pos_file = 'android/dev.pos.txt'
    dev_neg_file = 'android/dev.neg.txt'
    dev_data = dataset.read_annotations(dev_pos_file, dev_neg_file)

    test_pos_file = 'android/test.pos.txt'
    test_neg_file = 'android/test.neg.txt'
    test_data = dataset.read_annotations(test_pos_file, test_neg_file)

    dev_bm25_score = bm25_auc(dev_data, dataset)
    dev_tfidf_score = tfidf_auc(dev_data, dataset)
    print 'BM25: {}, TFIDF: {}'.format(dev_bm25_score, dev_tfidf_score)

    test_bm25_score = bm25_auc(test_data, dataset)
    test_tfidf_score = tfidf_auc(test_data, dataset)
    print 'BM25: {}, TFIDF: {}'.format(test_bm25_score, test_tfidf_score)


if __name__ == '__main__':
    main()
