import random
import numpy as np


def generate_classifier_train_batches(
    source_corpus_ids, target_corpus_ids,
        batch_size, n_batches, padding_id):

    source_ids = source_corpus_ids.values()
    target_ids = target_corpus_ids.values()
    batches = []
    labels = []
    # per_batch = batch_size / 2
    for i in range(n_batches):
        source_batch = random.sample(source_ids, batch_size)
        target_batch = random.sample(target_ids, batch_size)
        titles, bodies = zip(*(source_batch + target_batch))
        titles = list(titles)
        bodies = list(bodies)
        titles, bodies = create_batch(titles, bodies, padding_id)
        labels = np.array([0] * batch_size + [1] * batch_size)
        batches.append((titles, bodies, labels))
    return batches


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
