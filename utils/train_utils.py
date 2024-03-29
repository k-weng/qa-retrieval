import time

import torch
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
                            batches, padding_id, epoch, lmbda):
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
            torch.from_numpy(labels).type(torch.LongTensor))

        _, classes = torch.max(predictions.cpu().data, 1)
        total = len(labels)
        correct = (classes == labels.data).sum()
        accuracy_classifier = 100 * correct / float(total)

        if cuda_available:
            target = target.cuda()
            labels = labels.cuda()

        loss_encoder = criterion_encoder(scores, target)
        loss_classifier = criterion_classifier(predictions, labels)

        loss = loss_encoder - lmbda * loss_classifier
        loss_val = loss.cpu().data.numpy()[0]
        total_loss += loss_val
        total_encoder_loss += loss_encoder.cpu().data.numpy()[0]
        total_classifier_loss += loss_classifier.cpu().data.numpy()[0]

        loss.backward()
        optimizer_encoder.step()
        optimizer_classifier.step()

        print ('Epoch: {}/{}, {}/{}, ' +
               'Acc: {}, Loss: {}, Avg Loss: {}, ' +
               'Avg Encoder Loss: {}, Avg Classifier Loss: {}').format(
            epoch + 1, args.epochs, i + 1, len(batches),
            accuracy_classifier, loss_val, total_loss / (i + 1),
            total_encoder_loss / (i + 1), total_classifier_loss / (i + 1))


def train_adda(args, encoder_src, encoder_tgt, model_discrim, embedding,
               optimizer_tgt, optimizer_discrim, criterion,
               batches, padding_id, epoch):

    total_encoder_tgt_loss = 0.0
    total_discrim_loss = 0.0
    model_discrim.train()
    encoder_tgt.train()

    for i, batch in enumerate(batches):
        optimizer_discrim.zero_grad()

        title_ids, body_ids, labels = batch

        title_ids_src = title_ids[:, :args.batch_size]
        body_ids_src = body_ids[:, :args.batch_size]
        hidden_src = forward(
            args, encoder_src, embedding,
            title_ids_src, body_ids_src, padding_id)

        title_ids_tgt = title_ids[:, args.batch_size:]
        body_ids_tgt = body_ids[:, args.batch_size:]
        hidden_tgt = forward(
            args, encoder_tgt, embedding,
            title_ids_tgt, body_ids_tgt, padding_id)

        # hidden_all = torch.cat((hidden_src, hidden_tgt), 0)
        # predictions = model_discrim(hidden_all.detach())
        preds_src = model_discrim(hidden_src.detach())
        preds_tgt = model_discrim(hidden_tgt.detach())
        predictions = torch.cat((preds_src, preds_tgt), 0)
        labels_all = Variable(
            torch.from_numpy(labels).type(torch.LongTensor))
        if cuda_available:
            labels_all = labels_all.cuda()

        loss_discrim = criterion(predictions, labels_all)
        total_discrim_loss += loss_discrim.cpu().data.numpy()[0]

        loss_discrim.backward()

        optimizer_discrim.step()

        predictions_classes = torch.squeeze(predictions.max(1)[1])
        accuracy = (predictions_classes == labels_all).float().mean()
        accuracy = accuracy.cpu().data.numpy()[0]

        optimizer_discrim.zero_grad()
        optimizer_tgt.zero_grad()

        hidden_tgt = forward(
            args, encoder_tgt, embedding,
            title_ids_tgt, body_ids_tgt, padding_id)

        predictions_tgt = model_discrim(hidden_tgt)

        labels_tgt = Variable(
            1 - torch.from_numpy(
                labels[args.batch_size:]).type(torch.LongTensor))

        if cuda_available:
            labels_tgt = labels_tgt.cuda()

        loss_tgt = criterion(predictions_tgt, labels_tgt)
        total_encoder_tgt_loss += loss_tgt.cpu().data.numpy()[0]
        loss_tgt.backward()

        optimizer_tgt.step()
        optimizer_tgt.zero_grad()

        print ('Epoch: {}/{}, {}/{}, ' +
               'Acc: {}, Avg Discriminator Loss: {}, ' +
               'Avg Target Loss: {}').format(
            epoch + 1, args.epochs, i + 1, len(batches),
            accuracy, total_discrim_loss / (i + 1),
            total_encoder_tgt_loss / (i + 1))


def forward(args, encoder, embedding, title_ids, body_ids, padding_id):
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

    h_t = encoder(x_t)
    h_b = encoder(x_b)

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
