import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from metrics import Metrics
from meter import AUCMeter

cuda_available = torch.cuda.is_available()


def train(args, model, embedding, optimizer, criterion,
          batches, padding_id, epoch):
    total_loss = 0.0

    model.train()

    for i, batch in enumerate(batches):
        start = time.time()
        optimizer.zero_grad()

        # title_ids = title x questions
        # body_ids = body x questions
        # set_ids = pairs x sample (= 22)
        title_ids, body_ids, set_ids = batch
        n_pairs, sample_size = set_ids.shape

        # hidden = questions x hidden
        hidden = forward(args, model, embedding,
                         title_ids, body_ids, padding_id)

        # questions = pairs x sample (= 22) x hidden
        questions = hidden[set_ids.ravel()]
        questions = questions.view(n_pairs, sample_size, args.hidden)

        # q = pairs x 1 x hidden (= 200)
        # p = pairs x sample - 1 (= 21) x hidden
        q = questions[:, 0, :].unsqueeze(1)
        p = questions[:, 1:, :]

        # scores = pairs x sample - 1 (= 21)
        # target = pairs
        # scores = scoring(q, p)
        scores = F.cosine_similarity(q, p, dim=2)
        target = Variable(torch.zeros(n_pairs).type(torch.LongTensor))

        if cuda_available:
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


def train_encoder_classifer(args, model_encoder, model_classifier, embedding,
                            optimizer_encoder, optimizer_classifier,
                            criterion_encoder, criterion_classifier,
                            batches, padding_id, epoch, llambda):
    total_loss = 0.0
    total_encoder_loss = 0.0
    total_classifier_loss = 0.0
    model_encoder.train()
    model_classifier.train()

    for i, batch in enumerate(batches):
        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()

        batch_encoder, batch_classifier = batch
        encoder_title_ids, encoder_body_ids, set_ids = batch_encoder
        classifier_title_ids, classifier_body_ids, labels = batch_classifier
        n_pairs, sample_size = set_ids.shape

        hidden_encoder = forward(
            args, model_encoder, embedding,
            encoder_title_ids, encoder_body_ids, padding_id)

        questions = hidden_encoder[set_ids.ravel()]
        questions = questions.view(n_pairs, sample_size, args.hidden)

        q = questions[:, 0, :].unsqueeze(1)
        p = questions[:, 1:, :]

        scores = F.cosine_similarity(q, p, dim=2)
        target = Variable(torch.zeros(n_pairs).type(torch.LongTensor))

        hidden_classifier = forward(
            args, model_encoder, embedding,
            classifier_title_ids, classifier_body_ids, padding_id)
        predictions = model_classifier(hidden_classifier)
        labels = Variable(
            torch.from_numpy(labels).type(torch.FloatTensor))

        if cuda_available:
            target = target.cuda()
            labels = labels.cuda()

        loss_encoder = criterion_encoder(scores, target)
        loss_classifier = criterion_classifier(predictions, labels)

        loss = loss_encoder - llambda * loss_classifier
        loss_val = loss.cpu().data.numpy()[0]
        total_loss += loss_val
        total_encoder_loss += loss_encoder.cpu().data.numpy()[0]
        total_classifier_loss += loss_classifier.cpu().data.numpy()[0]

        loss.backward()
        optimizer_encoder.step()
        optimizer_classifier.step()

        print ('Epoch: {}/{}, Batch {}/{}, ' +
               'Loss: {}, Average Loss: {}, ' +
               'Average Encoder Loss: {}, ' +
               'Average Classifier Loss: {}').format(
            epoch + 1, args.epochs, i + 1, len(batches),
            loss_val, total_loss / (i + 1),
            total_encoder_loss / (i + 1), total_classifier_loss / (i + 1))


def forward(args, model, embedding, title_ids, body_ids, padding_id):
    embed_size = args.embed

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

    if cuda_available:
        x_t = x_t.cuda()
        x_b = x_b.cuda()

    h_t = model(x_t)
    h_b = model(x_b)

    # h_t = title x questions x hidden
    # h_b = body x questions x hidden
    h_t = normalize(h_t, 3)
    h_b = normalize(h_b, 3)

    # h_t = questions x hidden
    # h_b = questions x hidden
    h_t = average(h_t, title_ids, padding_id)
    h_b = average(h_b, body_ids, padding_id)

    # hidden = questions x hidden
    hidden = (0.5 * (h_t + h_b))
    hidden = normalize(hidden, 2)

    return hidden


def evaluate_metrics(args, model, embedding, batches, padding_id):
    model.eval()
    results = []

    for i, batch in enumerate(batches):
        # title_ids = title x questions (= 21)
        # body_ids = body x questions (= 21)
        # labels = questions - query (= 20)
        title_ids, body_ids, labels = batch

        # hidden = questions (= 21) x hidden
        hidden = forward(args, model, embedding,
                         title_ids, body_ids, padding_id)

        # q = 1 x hidden (= 200)
        # p = questions - query (= 20) x hidden
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


def evaluate_auc(args, model, embedding, batches, padding_id):
    model.eval()
    meter = AUCMeter()

    for i, batch in enumerate(batches):
        title_ids, body_ids, labels = batch

        hidden = forward(args, model, embedding,
                         title_ids, body_ids, padding_id)

        q = hidden[0].unsqueeze(0)
        p = hidden[1:]

        scores = F.cosine_similarity(q, p, dim=1).cpu().data
        assert len(scores) == len(labels)

        target = torch.DoubleTensor(labels)
        meter.add(scores, target)

    auc_score = meter.value(0.05)

    print 'AUC(0.05): {}'.format(auc_score)
    return auc_score


def normalize(hidden, dim, eps=1e-8):
    assert dim in [2, 3]
    return hidden / (torch.norm(hidden, 1, dim - 1, keepdim=True) + eps)


def average(hidden, ids, padding_id, eps=1e-8):
    # mask = sequence x questions x 1
    mask = Variable(torch.from_numpy(1 * (ids != padding_id))
                    .type(torch.FloatTensor).unsqueeze(2))

    if cuda_available:
        mask = mask.cuda()

    # masked_sum = questions x hidden
    masked_sum = torch.sum(mask * hidden, dim=0)

    # lengths = questions x 1
    lengths = torch.sum(mask, dim=0)

    return masked_sum / (lengths + eps)


class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
