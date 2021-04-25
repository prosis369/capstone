import numpy as np
import torch
from torch import nn
from torch import optim
from utils import np2autograd
from torch.autograd import Variable
from torch.nn import functional as F
from utils import max_sentence_size
from utils import avg_cross_entropy_loss
from utils import batch_generator_bias
from sklearn.metrics import accuracy_score

dataset_file = './olid-train.csv'

# Hyperparams
postag_reg = 1e-3
chunking_reg = 1e-3
learning_rate = 1e-3
postag_hn_size = 100
postag_nb_layers = 2
embedding_size = 512
bias_reg = 1e-3
nb_postags = 1

class TransferBiasClassification(nn.Module):
  
    def __init__(self):
        super(TransferBiasClassification, self).__init__()

        self.w = nn.Parameter(torch.randn(postag_nb_layers * 2,
                                          max_sentence_size,
                                          postag_hn_size))
        self.h = nn.Parameter(torch.randn(postag_nb_layers * 2,
                                          max_sentence_size,
                                          postag_hn_size))

        # Bidirectional LSTM
        self.bi_lstm = nn.LSTM(embedding_size,
                               postag_hn_size,
                               postag_nb_layers,
                               bidirectional=True)

        self.fc = nn.Linear(postag_hn_size * 2, nb_postags)

    def forward(self, x):
        # Runs the LSTM for each word-vector in the sentence x
        # x = [x]
        x = torch.tensor(x, dtype=torch.float32)
        out, hn = self.bi_lstm(x, (self.h[:, :x.size(1), :],
                                   self.w[:, :x.size(1), :]))

        # Runs a linear classifier on the outputed state vector
        tags = self.fc(out[0])

        return tags
    
    def bias_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (transfer_bias_model.w.norm() ** 2) * bias_reg

        return loss

    def loss(self, y, bias):
        # print("Hi i am in loss")
        losses = []
        # print("Hi i am in loss line 2")
        length = len(y)

        for i in range(length):

            p_bias, r_bias = y[i][0], np2autograd(bias[i])
            loss_bias = self.bias_loss(p_bias, r_bias)

            losses.append(loss_bias)

        loss = losses[0]

        for i in range(1, length):
            loss += losses[i]

        loss = loss / length

        return loss


def threshold(prediction, upperBound, lowerBound):
  if prediction >= upperBound:
      prediction = 1
  elif prediction < lowerBound:
      prediction = -1
  else:
      prediction = 0
  return prediction

def compare(out):

    # print(out[0][0].numpy())
    predicted_bias = out[0][0].numpy()
    predicted_bias = threshold(predicted_bias, 0.5, 0)
    
    return [predicted_bias]

def accuracy(train_batch_acc, bias_nb_batches):

  pred_bias = []

  l = len(train_batch_acc)

  for i in range(l):
    pred_bias.append(train_batch_acc[i][0])

  print(pred_bias)
  print(bias_nb_batches)

  bias_acc = accuracy_score(bias_nb_batches, pred_bias)
  
  return(bias_acc)

nb_epochs = 1
# batch_size = 47
batch_size = 1
nb_batches = 1956

gen = batch_generator_bias(batch_size, nb_batches, dataset_file)
transfer_bias_model = TransferBiasClassification()
adam = optim.Adam(transfer_bias_model.parameters(), lr=learning_rate)

train_epoch_acc = []
PATH = "saved_model_bias.pt"


for epoch in range(nb_epochs):

    train_batch_loss = []
    train_batch_acc = []

    transfer_bias_nb_batches = []

    for batch in range(nb_batches):

        text, transfer_bias = next(gen)

        transfer_bias_nb_batches.append(transfer_bias[0][0])
        
        out = transfer_bias_model.forward(text)

        loss = transfer_bias_model.loss(out, transfer_bias)
        print("Epoch:", epoch,
              "Batch:", batch,
              "Loss:", loss.data[0])

        adam.zero_grad()
        # loss.backward()
        loss.sum().backward()
        adam.step()
        # print("out", out)

        torch.save(transfer_bias_model.state_dict(), PATH)


        transfer_bias_model.eval() # enter evaluation mode
        with torch.no_grad():
              train_batch_acc.append(compare(out)) # evaluate mini-batch train accuracy in evaluation

    acc = accuracy(train_batch_acc, transfer_bias_nb_batches)
    print("Epoch: ", epoch, "Bias Accuracy: ", acc[0])

