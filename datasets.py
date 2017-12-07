import gzip
import random


class Dataset(object):
    def __init__(self, file):
        corpus = {}

        with gzip.open(file) as f:
            for line in f:
                qid, title, body = line.split('\t')
                if len(title) == 0:
                    continue
                title = title.strip().split()
                body = body.strip().split()
                corpus[qid] = (title, body)

        print 'Corpus processed.'
        self.corpus = corpus

    def get_corpus(self):
        return self.corpus


class UbuntuDataset(Dataset):
    def __init__(self, file):
        super(UbuntuDataset, self).__init__(file)

    def read_annotations(self, file, max_neg=20):
        data = []
        with open(file) as f:
            for line in f:
                qid, pos, neg = line.split('\t')[:3]
                pos = pos.split()
                neg = neg.split()

                if max_neg != -1:
                    random.shuffle(neg)
                    neg = neg[:max_neg]

                ids = neg + pos

                seen = set()
                pids = []
                labels = []
                for id in ids:
                    if id not in seen:
                        pids.append(id)
                        labels.append(0 if id not in pos else 1)
                        seen.add(id)
                data.append((qid, pids, labels))
        return data
