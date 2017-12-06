import preprocessing
from embedding import Embedding
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import sys
import os
import argparse
import shutil
from metrics import Metrics

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--embed', type=int, default=200)
parser.add_argument('--lr', '-lr', type=float, default=0.001)
parser.add_argument('--batch_size', '-b', type=int, default=40)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', '-e', type=int, default=50)
parser.add_argument('--margin', '-m', type=float, default=0.5)
parser.add_argument('--resume', '-r', type=str, default='')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--test', action='store_true')

best_mrr = -1


def main():
    global args, best_mrr
    args = parser.parse_args()

    embed_size = args.embed
    hidden_size = args.hidden
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    margin = args.margin

    corpus_file = 'askubuntu/text_tokenized.txt.gz'
    corpus = preprocessing.read_corpus(corpus_file)
    print 'Corpus processed.'

    embedding_file = 'askubuntu/vector/vectors_pruned.200.txt.gz'
    embedding = Embedding(embed_size, embedding_file)
    print 'Embeddings loaded.'

    corpus_ids = embedding.corpus_to_ids(corpus)
    padding_id = embedding.vocab_ids['<padding>']

    train_file = 'askubuntu/train_random.txt'
    train_data = preprocessing.read_annotations(train_file)

    dev_file = 'askubuntu/dev.txt'
    dev_data = preprocessing.read_annotations(dev_file, max_neg=-1)
    dev_batches = preprocessing.generate_eval_batches(
        corpus_ids, dev_data, padding_id)

    model = nn.LSTM(embedding.embed_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.MultiMarginLoss(margin=margin)
    print 'Model created.'

    if os.path.isfile(args.resume):
        print 'Loading checkpoint.'
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_mrr = checkpoint['best_mrr']
        model.load_state_dict(checkpoint['state_dict'])

        print 'Loaded checkpoint at epoch {}.'.format(checkpoint['epoch'])
    else:
        print 'No checkpoint found here.'

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in xrange(args.start_epoch, epochs):
        train_batches = preprocessing.generate_train_batches(
            corpus_ids, train_data, batch_size, padding_id)

        train(model, embedding, optimizer, criterion,
              train_batches, padding_id, epoch)

        map, mrr, p1, p5 = evaluate(
            model, embedding, dev_batches, padding_id)

        is_best = mrr > best_mrr
        best_mrr = max(mrr, best_mrr)
        save({
            'epoch': epoch + 1,
            'arch': 'lstm',
            'state_dict': model.state_dict(),
            'best_mrr': best_mrr,
        }, is_best, args.hidden)


def train(model, embedding, optimizer, criterion, batches, padding_id, epoch):
    total_loss = 0.0

    model.train()

    for i, batch in enumerate(batches):
        start = time.time()
        optimizer.zero_grad()

        # title_ids = title x questions
        # body_ids = body (= 100) x questions
        # set_ids = pairs x sample (= 22)
        title_ids, body_ids, set_ids = batch
        n_pairs, sample_size = set_ids.shape

        # hidden = questions x hidden
        hidden = forward(model, embedding,
                         title_ids, body_ids, padding_id)

        # questions = pairs x sample (= 22) x hidden (= 200)
        questions = hidden[set_ids.ravel()]
        questions = questions.view(n_pairs, sample_size, args.hidden)

        # q = pairs x 1 x hidden (= 200)
        # p = pairs x sample - 1 (= 21) x hidden (= 200)
        q = questions[:, 0, :].unsqueeze(1)
        p = questions[:, 1:, :]

        # scores = pairs x sample - 1 (= 21)
        # target = pairs
        # scores = scoring(q, p)
        scores = F.cosine_similarity(q, p, dim=2)
        target = Variable(torch.zeros(n_pairs).type(torch.LongTensor))

        if args.cuda:
            target = target.cuda()

        loss = criterion(scores, target)

        loss_val = loss.cpu().data.numpy()[0]
        total_loss += loss_val

        loss.backward()
        optimizer.step()

        print ('Epoch: {}/{}, Batch {}/{}, Time: {}, ' +
               'Loss: {}, Average Loss: {}').format(
            epoch + 1, args.epochs, i + 1, len(batches), time.time() - start,
            loss_val, total_loss / (i + 1))


def evaluate(model, embedding, batches, padding_id):
    model.eval()
    results = []
    for i, batch in enumerate(batches):
        # title_ids = title x questions (= 21)
        # body_ids = body (< 100) x questions (= 21)
        # labels = questions - query (= 20)
        title_ids, body_ids, labels = batch

        # hidden = questions (= 21) x hidden (= 200)
        hidden = forward(model, embedding,
                         title_ids, body_ids, padding_id)

        # q = 1 x hidden (= 200)
        # p = questions - query (= 20) x hidden (= 200)
        q = hidden[0].unsqueeze(0)
        p = hidden[1:]

        scores = F.cosine_similarity(q, p, dim=1).cpu().data.numpy()
        assert len(scores) == len(labels)

        ranking = (-1 * scores).argsort()
        results.append(labels[ranking])

    metrics = Metrics(results)

    map = metrics.map() * 100
    mrr = metrics.mrr() * 100
    p1 = metrics.precision(1) * 100
    p5 = metrics.precision(5) * 100

    print 'MAP: {}, MRR: {}, P@1: {}, P@5: {}'.format(
        map, mrr, p1, p5)

    return map, mrr, p1, p5


def forward(model, embedding, title_ids, body_ids, padding_id):
    embed_size = args.embed
    hidden_size = args.hidden

    assert title_ids.shape[1] == body_ids.shape[1]
    title_len, n_questions = title_ids.shape
    body_len = body_ids.shape[0]

    # x_t = title x questions x embed (= 200)
    x_t = embedding.get_embeddings(title_ids.ravel())
    x_t = torch.from_numpy(x_t).type(torch.FloatTensor)
    x_t = x_t.view(title_len, n_questions, embed_size)

    # x_b = body (= 100) x questions x embed (= 200)
    x_b = embedding.get_embeddings(body_ids.ravel())
    x_b = torch.from_numpy(x_b).type(torch.FloatTensor)
    x_b = x_b.view(body_len, n_questions, embed_size)

    x_t = Variable(x_t)
    x_b = Variable(x_b)

    if args.cuda:
        x_t = x_t.cuda()
        x_b = x_b.cuda()

    # h_c_t[0] = 1 x questions x hidden (= 200)
    # h_c_b[0] = 1 x questions x hidden (= 200)
    h_c_t = (Variable(torch.zeros(1, n_questions, hidden_size)),
             Variable(torch.zeros(1, n_questions, hidden_size)))
    h_c_b = (Variable(torch.zeros(1, n_questions, hidden_size)),
             Variable(torch.zeros(1, n_questions, hidden_size)))

    # h_t = title x questions x hidden (= 200)
    # h_b = body x questions x hidden (= 200)
    h_t = Variable(torch.zeros(title_len, n_questions, hidden_size))
    h_b = Variable(torch.zeros(body_len, n_questions, hidden_size))

    if args.cuda:
        h_c_t = (h_c_t[0].cuda(), h_c_t[1].cuda())
        h_c_b = (h_c_b[0].cuda(), h_c_b[1].cuda())
        h_t = h_t.cuda()
        h_b = h_b.cuda()

    # h_t = title x questions x hidden (= 200)
    for j in xrange(title_len):
        _, h_c_t = model(x_t[j].view(1, n_questions, -1), h_c_t)
        h_t[j, :, :] = h_c_t[0]

    # h_b = body (= 100) x questions x hidden (= 200)
    for j in xrange(body_len):
        _, h_c_b = model(x_b[j].view(1, n_questions, -1), h_c_b)
        h_b[j, :, :] = h_c_b[0]

    # h_t = title x questions x hidden (= 200)
    # h_b = body (= 100) x questions x hidden (= 200)
    h_t = normalize(h_t, 3)
    h_b = normalize(h_b, 3)

    # h_t = questions x hidden (= 200)
    # h_b = questions x hidden (= 200)
    h_t = average(h_t, title_ids, padding_id)
    h_b = average(h_b, body_ids, padding_id)

    # hidden = questions x hidden
    hidden = (0.5 * (h_t + h_b))
    hidden = normalize(hidden, 2)

    return hidden


def normalize(hidden, dim, eps=1e-8):
    assert dim in [2, 3]
    return hidden / (torch.norm(hidden, 1, dim - 1, keepdim=True) + eps)


def average(hidden, ids, padding_id, eps=1e-8):
    # mask = sequence (title or body) x questions x 1
    mask = Variable(torch.from_numpy(1 * (ids != padding_id))
                    .type(torch.FloatTensor).unsqueeze(2))

    if args.cuda:
        mask = mask.cuda()

    # masked_sum = questions x hidden (= 200)
    masked_sum = torch.sum(mask * hidden, dim=0)

    # lengths = questions x 1
    lengths = torch.sum(mask, dim=0)

    return masked_sum / (lengths + eps)


def save(state, is_best, hidden):
    latest = 'lstm_{}_latest.pth.tar'.format(hidden)
    torch.save(state, latest)
    if is_best:
        best = 'lstm_{}_best.pth.tar'.format(hidden)
        shutil.copyfile(latest, best)


if __name__ == '__main__':
    main()
