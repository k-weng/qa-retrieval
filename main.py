import preprocessing
from embedding import Embedding
from torch.autograd import Variable
import torch.nn as nn
import torch
import time


def main():
    embed_size = 200
    hidden_size = 200
    batch_size = 40
    lr = 0.001
    epochs = 50
    margin = 0.5

    corpus_file = 'askubuntu/text_tokenized.txt.gz'
    embedding_file = 'askubuntu/vector/vectors_pruned.200.txt.gz'
    train_file = 'askubuntu/train_random.txt'

    corpus = preprocessing.read_corpus(corpus_file)
    embedding = Embedding(embed_size, embedding_file)
    train = preprocessing.read_annotations(train_file)

    corpus_ids = embedding.corpus_to_ids(corpus)
    padding_id = embedding.vocab_ids['<padding>']

    lstm = nn.LSTM(embedding.embed_size, hidden_size)
    optimizer = torch.optim.Adam(lstm.parameters(), lr)
    scoring = nn.CosineSimilarity(dim=2)
    criterion = nn.MultiMarginLoss(margin=margin)
    print 'Model created.'

    for epoch in xrange(epochs):
        batches = preprocessing.generate_batches(
            corpus_ids, train, batch_size, padding_id)
        total_loss = 0.0

        for i, batch in enumerate(batches):
            start = time.time()
            optimizer.zero_grad()

            # title_ids = title x questions
            # body_ids = body (= 100) x questions
            # set_ids = pairs x sample (= 22)
            title_ids, body_ids, set_ids = batch

            assert title_ids.shape[1] == body_ids.shape[1]
            title_len, n_questions = title_ids.shape
            body_len = body_ids.shape[0]
            n_pairs, sample_size = set_ids.shape

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

            # h_t = title x questions x hidden (= 200)
            for j in xrange(title_len):
                _, h_c_t = lstm(x_t[j].view(1, n_questions, -1), h_c_t)
                h_t[j, :, :] = h_c_t[0]

            # h_b = body (= 100) x questions x hidden (= 200)
            for j in xrange(body_len):
                _, h_c_b = lstm(x_b[j].view(1, n_questions, -1), h_c_b)
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

            # questions = pairs x sample (= 22) x hidden (= 200)
            questions = hidden[set_ids.ravel()]
            questions = questions.view(n_pairs, sample_size, hidden_size)

            # p = pairs x 1 x hidden (= 200)
            # q = pairs x sample - 1 (= 21) x hidden (= 200)
            q = questions[:, 0, :].unsqueeze(1)
            p = questions[:, 1:, :]

            # scores = pairs x sample - 1 (= 21)
            # target = pairs
            scores = scoring(q, p)
            target = Variable(torch.zeros(n_pairs).type(torch.LongTensor))
            loss = criterion(scores, target)
            total_loss += loss.data.numpy()[0]

            loss.backward()
            optimizer.step()

            print ('Epoch: {0}/{1}, Batch {2}/{3}, Time: {4}, ' +
                   'Loss: {5}, Average Loss: {6}').format(
                epoch + 1, epochs, i + 1, len(batches), time.time() - start,
                loss.data.numpy()[0], total_loss / (i + 1))


def normalize(hidden, dim, eps=1e-8):
    assert dim in [2, 3]
    return hidden / (torch.norm(hidden, 1, dim - 1, keepdim=True) + eps)


def average(hidden, ids, padding_id, eps=1e-8):
    # mask = sequence (title or body) x questions x 1
    mask = Variable(torch.from_numpy(1 * (ids != padding_id))
                    .type(torch.FloatTensor).unsqueeze(2))

    # masked_sum = questions x hidden (= 200)
    masked_sum = torch.sum(mask * hidden, dim=0)

    # lengths = questions x 1
    lengths = torch.sum(mask, dim=0)

    return masked_sum / (lengths + eps)


if __name__ == '__main__':
    main()
