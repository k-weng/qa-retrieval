from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import torch.nn as nn
import torch
import numpy as np

train_ids = open("../data/train_random.txt")
question_examples_temp = train_ids.read().split('\n')
question_examples = []
for q in question_examples_temp[:15]:
	array = q.split("\t")
	positive = array[1].split(" ")
	p = int(array[0])
	for p_plus in positive:
		q_minus = [int(x) for x in array[2].split(" ")][:20]
		question_examples.append([p,int(p_plus), q_minus])#pi, pi+, Q-
train_ids.close()

question_ids_file = open("../data/text_tokenized.txt")
question_ids = question_ids_file.read().split('\n')
question_ids_file.close()

qids_title = {} #maps question ids to question titles
qids_body = {} #maps question ids to question bodies
for q in question_ids:
	id_num = q.split("\t")[0]
	if id_num != "":
		id_num = int(id_num)
		qids_title[id_num] = q.split("\t")[1]#title only
		qids_body[id_num] = q.split("\t")[2]#title only

word_ids_file = open("../data/vectors_pruned.200.txt")
word_ids = word_ids_file.read().split('\n')
word_ids_file.close()
wids = {}
for w in word_ids:
	id_num = w.split(" ")[0]
	arr = w.split(" ")[1:-1]
	arr = [float(x) for x in arr]
	wids[id_num] = arr

title_batches = []
body_batches = []

print(len(question_examples))

count = 0
title_batch = []
body_batch = []
batch_size = 5
for ex in question_examples:
	p, p_plus, q_minus = ex[0], ex[1], ex[2]
	questions = [p,p_plus]+ex[2]
	training_example_title = [] #22 question titles
	training_example_body = []
	for i in range(len(questions)):
		id_num = questions[i]
		question_title = qids_title[id_num].split(" ")
		question_body = qids_body[id_num].split(" ")
		training_example_title.append(question_title)
		training_example_body.append(question_body)
	title_batch.append(training_example_title)
	body_batch.append(training_example_body)
	count += 1
	if count == batch_size:
		print(len(title_batches))
		title_batches.append(title_batch)
		body_batches.append(body_batch)
		body_batch = []
		title_batch = []
		count = 0
print("batches")
print(len(title_batches))

def pad_batch(batches):
	temp_batches = []
	orig_len_batches = []
	for i in range(len(batches)):
		batch = batches[i]
		max_len = 0
		for training_example in batch:
			for question in training_example:
				max_len = max(max_len, len(question))
		temp_batch = []
		for j in range(len(batch)):
			training_example = batch[j]
			temp_training_example = []
			for k in range(len(training_example)):
				question = training_example[k]
				pad_len = max_len - len(question)
				padding = ["UNK"]*pad_len
				temp_question = question + padding
				temp_training_example.append((temp_question, len(question)))
			temp_batch.append(temp_training_example)
		temp_batches.append(temp_batch)
		print(i)
	return temp_batches

print("title padding")
title_batches = pad_batch(title_batches)
print("body padding")
body_batches = pad_batch(body_batches)

embed_dim = 200
hidden_dim = 100
kernel_size = 3

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.c1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
	def forward(self, x):
		c = self.c1(x)
		out = F.tanh(c)
		return out
		
cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=0.001)
loss_function = torch.nn.MultiMarginLoss(p=1, margin=1, size_average=True)

def get_embed_seq(arr):
	res = []
	for word in arr:
		w = [0]*200
		if word in wids and wids[word] != []:
			w = wids[word]
		res.append(w)
	return res

def forward_training_example(training_examples):
	title_input = [get_embed_seq(training_examples[i][0]) for i in range(len(training_examples))]
	title_input = Variable(torch.FloatTensor(title_input))
	title_input = torch.transpose(title_input,2,1) #(batch_size x input_size x seq_len) for CNN
	title_outputs = cnn(title_input)
	length = title_outputs.size()[2]
	title_outputs = title_outputs.sum(2) #sum across sequence length
	title_lens = torch.FloatTensor([training_examples[i][1] for i in range(len(training_examples))]) 
	title_lens = Variable(title_lens).unsqueeze(1).expand_as(title_outputs)
	title_outputs = title_outputs/title_lens
	return title_outputs

num_epochs = 1
for epoch in range(num_epochs):
	for i in range(len(body_batches)):
		title_batch = title_batches[i]
		body_batch = body_batches[i]
		print(title_batch)
		for training_titles, training_bodies in zip(title_batch, body_batch): #one training example
			optimizer.zero_grad()
			title_outputs = forward_training_example(training_titles)
			body_outputs = forward_training_example(training_bodies)

			outputs = (body_outputs + title_outputs)/2

			y = Variable(torch.zeros(1).type(torch.LongTensor)) #[batch_size]
			q = outputs[0].unsqueeze(0)
			p = outputs[1:]
			cos_similar = F.cosine_similarity(q,p,1)
			loss = loss_function(cos_similar, y)
			loss.backward()
			print(loss)
			optimizer.step()
			# output = np.array([2,2])
			# loss = loss_function(output)
			# p, p_plus, q_minus = ex[0], ex[1], ex[2]
			# print(p, p_plus, q_minus)
