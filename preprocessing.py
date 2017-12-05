import gzip
import random
import numpy as np


def read_corpus(file):
    corpus = {}

    with gzip.open(file) as f:
        for line in f:
            qid, title, body = line.split('\t')
            if len(title) == 0:
                continue
            title = title.strip().split()
            body = body.strip().split()
            corpus[qid] = (title, body)

    return corpus


def read_annotations(file, max_neg=20):
    data = []
    with open(file) as f:
        for line in f:
            qid, pos, neg = line.split('\t')[:3]
            pos = pos.split()
            neg = neg.split()

            if max_neg != -1:
                random.shuffle(neg)
                neg = neg[:max_neg]

            ids = pos + neg

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


def generate_train_batches(corpus_ids, data, batch_size, padding_id):
    perm = range(len(data))
    random.shuffle(perm)

    batch_count = 0
    n_qids = len(data)
    batches = []

    local_ids = {}
    titles, bodies, sets = [], [], []

    for i in xrange(n_qids):
        qid, pids, labels = data[perm[i]]
        if qid not in corpus_ids:
            continue
        batch_count += 1
        for id in [qid] + pids:
            if id not in local_ids:
                if id not in corpus_ids:
                    continue
                local_ids[id] = len(titles)
                title, body = corpus_ids[id]
                titles.append(title)
                bodies.append(body)

        local_qid = local_ids[qid]

        pos, neg = [], []
        for pid, label in zip(pids, labels):
            if pid in local_ids:
                pos.append(local_ids[pid]) \
                    if label == 1 else neg.append(local_ids[pid])

        sets.extend([[local_qid, id] + neg for id in pos])

        if batch_count == batch_size or i == n_qids:
            batches.append(create_batch(titles, bodies, padding_id, sets=sets))
            batch_count = 0
            local_ids = {}
            titles, bodies, sets = [], [], []

    return batches


def generate_eval_batches(corpus_ids, data, padding_id):
    batches = []
    for qid, pids, labels in data:
        titles = []
        bodies = []
        for id in [qid] + pids:
            title, body = corpus_ids[id]
            titles.append(title)
            bodies.append(body)
        titles, bodies = create_batch(titles, bodies, padding_id)
        batches.append((titles, bodies, np.array(labels)))
    return batches


def create_batch(titles, bodies, padding_id, sets=None):
    title_len = max(1, max(len(title) for title in titles))
    body_len = max(1, max(len(body) for body in bodies))
    titles = np.column_stack([np.pad(title, (0, title_len - len(title)),
                              'constant', constant_values=padding_id)
                              for title in titles])
    bodies = np.column_stack([np.pad(body, (0, body_len - len(body)),
                              'constant', constant_values=padding_id)
                              for body in bodies])

    if sets:
        set_len = max(len(q_set) for q_set in sets)
        sets = np.vstack([np.pad(q_set, (0, set_len - len(q_set)), 'edge')
                          for q_set in sets])
        return titles, bodies, sets
    else:
        return titles, bodies
