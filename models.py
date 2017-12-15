import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

cuda_available = torch.cuda.is_available()


class CNN(nn.Module):
    def __init__(self, embed, hidden):
        super(CNN, self).__init__()

        self.hidden = hidden
        self.embed = embed

        kernel = 3
        conv1d = nn.Conv1d(self.embed, self.hidden, kernel, padding=1)
        self.conv1d = conv1d.cuda() if cuda_available else conv1d

    def forward(self, input):
        # input = sequence x questions x embed
        assert self.embed == input.size(2)

        # input = questions x embed x sequence
        input = input.transpose(0, 2).transpose(0, 1)

        # output = questions x hidden x sequence
        output = self.conv1d(input)
        output = F.tanh(output)

        # output = sequence x questions x hidden
        output = output.transpose(0, 1).transpose(0, 2)

        return output


class LSTM(nn.Module):
    def __init__(self, embed, hidden):
        super(LSTM, self).__init__()

        self.hidden = hidden
        self.embed = embed

        lstm = nn.LSTM(self.embed, self.hidden)
        self.lstm = lstm.cuda() if cuda_available else lstm

    def forward(self, input):
        # input = sequence x questions x embed
        seq_len, n_questions = input.size(0), input.size(1)
        assert self.embed == input.size(2)

        # h_c[0] = 1 x questions x hidden
        h_c = (Variable(torch.zeros(1, n_questions, self.hidden)),
               Variable(torch.zeros(1, n_questions, self.hidden)))

        # output = sequence x questions x hidden
        output = Variable(torch.zeros(seq_len, n_questions, self.hidden))

        if cuda_available:
            h_c = (h_c[0].cuda(), h_c[1].cuda())
            output = output.cuda()

        # output = sequence x questions x hidden
        for j in xrange(seq_len):
            _, h_c = self.lstm(input[j].view(1, n_questions, -1), h_c)
            output[j, :, :] = h_c[0]

        return output


class FFN(nn.Module):
    def __init__(self, input, hidden1=300, hidden2=150):
        super(FFN, self).__init__()

        self.w1 = nn.Linear(input, hidden1)
        self.w2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.w1(input)
        x = F.relu(x)
        x = self.w2(x)
        x = F.relu(x)
        output = self.out(x)
        output = self.softmax(output)[:, 1]

        return output
