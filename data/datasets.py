import gzip
import random
from collections import defaultdict


class Dataset(object):
    def __init__(self, file):
        corpus = {}

        with gzip.open(file) as f:
            for line in f:
                qid, title, body = line.split('\t')
                if len(title) == 0:
                    continue

                title = [w.lower() for w in title.strip().split()]
                body = [w.lower() for w in body.strip().split()]

                corpus[qid] = (title, body)

                # corpus[qid] = (title.strip().split(),
                #                body.strip().split())

        print 'Corpus processed.'
        self.corpus = corpus

    def get_corpus(self):
        return self.corpus

    def retrieve_combined(self, query=None, joined=True):
        sequences = []
        if query:
            if isinstance(query, str):
                sequences.append(self.retrieve(query, joined, True))
            else:
                for id in query:
                    sequences.append(self.retrieve(id, joined, True))
        else:
            for id in self.corpus.keys():
                sequences.append(self.retrieve(id, joined, True))

        return sequences

    def retrieve_titles_bodies(self, query=None, joined=True):
        titles = []
        bodies = []
        if query:
            if isinstance(query, str):
                title, body = self.retrieve(query, joined, False)
                titles.append(title)
                bodies.append(body)
            else:
                for id in query:
                    title, body = self.retrieve(query, joined, False)
                    titles.append(title)
                    bodies.append(body)
        else:
            for id in self.corpus.keys():
                title, body = self.retrieve(query, joined, False)
                titles.append(title)
                bodies.append(body)

        return titles, bodies

    def retrieve(self, query, joined, together):
        title, body = self.corpus[query]
        if together:
            combined = title + body
            return ' '.join(combined) if joined else combined
        else:
            return ' '.join(title), ' '.join(body) if joined else title, body


class AndroidDataset(Dataset):
    def __init__(self, file):
        super(AndroidDataset, self).__init__(file)

    def read_annotations(self, pos_file, neg_file):
        pos_annotations = self.map_annotations(pos_file)
        neg_annoations = self.map_annotations(neg_file)

        ids = pos_annotations.keys()
        assert set(ids) == set(neg_annoations.keys())

        data = []
        for qid in ids:
            pos = pos_annotations[qid]
            neg = neg_annoations[qid]

            pids = neg[:]
            labels = [0] * len(neg)

            pids.extend(pos)
            labels.extend([1] * len(pos))

            data.append((qid, pids, labels))

        return data

    def map_annotations(self, file):
        annotations = defaultdict(list)

        with open(file) as f:
            for line in f:
                qid, pid = line.split()
                annotations[qid].append(pid)

        return annotations


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
