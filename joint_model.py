import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from utils import batch_generator
# from utils import nb_classes
# from utils import nb_postags
# from utils import nb_chunktags
from utils import max_sentence_size
# from utils import vec2word
from utils import np2autograd
# from lang_model import CharacterLanguageModel
# from lang_model import embedding_size
# from pos_tag import POSTag
# from chunking import Chunking
# from dependency import Dependency
from sentiment import SentimentClassification
from stance import StanceClassification
from sklearn.metrics import accuracy_score
# from predict import prediction

# Hyperparams
learning_rate = 1e-3
postag_reg = 1e-3
chunking_reg = 1e-3
sentiment_reg = 1e-3
stance_reg = 1e-3

class JointMultiTaskModel(nn.Module):
    def __init__(self, mode='all'):
        super(JointMultiTaskModel, self).__init__()

        # models
        # self.lang_model = CharacterLanguageModel()
        # self.postag = POSTag()
        # self.chunking = Chunking()
        # self.dependency = Dependency()
        self.sentiment = SentimentClassification()
        self.stance = StanceClassification()

        # Mode - all or module_name
        self.mode = mode

    # def embedded_batch(self, batch_x):
    #     embedded_batch = self.lang_model(batch_x)
    #
    #     for batch in embedded_batch:
    #         sent = np.zeros((1, len(batch), embedding_size), dtype=np.float32)
    #
    #         for i, word in enumerate(batch):
    #             sent[0, i] = word.data.numpy()
    #
    #         sent = torch.from_numpy(sent)
    #         sent = Variable(sent)
    #
    #         yield sent

    # def get_postag(self, sentence):
    #     return self.postag(sentence)
    #
    # def get_chunking(self, sentence, tags, h_tags):
    #     return self.chunking(sentence, tags, h_tags)
    #
    # def get_dependency(self, sentence, tags, h_tags, chunks, h_chunks):
    #     return self.dependency(sentence, tags, h_tags, chunks, h_chunks)
    #
    # def get_sentiment(self, sentence, tags, h_tags, chunks, h_chunks, deps, h_deps):
    #     return self.sentiment(sentence, tags, h_tags, chunks, h_chunks, deps, h_deps)

    def get_sentiment(self, sentence):
        return self.sentiment(sentence)

    def get_stance(self, sentence):
        return self.stance(sentence)

    def run_all(self, x):
        # sentence = self.embedded_batch(x)
        # print(x)
        for s in x:
            # print(len(s[0]))
            y_sentiment = self.get_sentiment(s)
            y_stance = self.get_stance(s)

            # y_chk, h_chk = self.get_chunking(s, y_pos, h_pos)
            # y_dep, h_dep = self.get_dependency(s, y_pos, h_pos, y_chk, h_chk)
            # y_sent, h_sent = self.get_sentiment(s, y_pos, h_pos, y_chk, h_chk, y_dep, h_dep)

            yield y_sentiment, y_stance

    def forward(self, x):
        # print(len(x))
        out = self.run_all(x)
        # print(out)
        out = list(out)

        return out

    # def postag_loss(self, y, yt):
    #     loss = F.cross_entropy(y, yt) \
    #            + (self.postag.w.norm() ** 2) * postag_reg
    #
    #     return loss
    #
    # def chunking_loss(self, y, yt):
    #     loss = F.cross_entropy(y, yt) \
    #            + (self.chunking.w.norm() ** 2) * chunking_reg
    #
    #     return loss

    def sentiment_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.sentiment.w.norm() ** 2) * sentiment_reg

        return loss

    def stance_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.stance.w.norm() ** 2) * stance_reg

        return loss

    def loss(self, y, sentiment, stance):
        # print("Hi i am in loss")
        losses = []
        # print("Hi i am in loss line 2")
        length = len(y)
        # print("Hi i am in loss line 3")
        # print(length)

        for i in range(length):

            # p_tag, r_tag = y[i][0], np2autograd(tags[i])
            # p_chunk, r_chunk = y[i][1], np2autograd(chunks[i])
            # p_sent, r_sent = y[i][3], np2autograd(sentiment[i])

            p_sent, r_sent = y[i][0], np2autograd(sentiment[i])
            p_stance, r_stance = y[i][1], np2autograd(stance[i])

            # loss_tag = self.postag_loss(p_tag, r_tag)
            # loss_chk = self.chunking_loss(p_chunk, r_chunk)

            loss_sent = self.sentiment_loss(p_sent, r_sent)
            loss_stance = self.stance_loss(p_stance, r_stance)

            # loss = loss_tag * 0.3 + loss_chk * 0.2 + loss_sent * 0.5
            # loss = loss / 3

            loss = loss_sent * 0.5 + loss_stance * 0.5
            loss = loss/2

            losses.append(loss)

        loss = losses[0]

        for i in range(1, length):
            loss += losses[i]

        loss = loss / length

        return loss

def compare(out):

    # print(out)
    # actual_sent = actual_sent[0][0]
    # actual_stance = actual_stance[0][0]
    # print(out[0][0].detach())
    # print(out[0][1].detach())

    # print(out[0][0].numpy()[0][0])
    predicted_sent = out[0][0].numpy()[0][0]
    predicted_stance = out[0][1].numpy()[0][0]

    if predicted_sent >= 0.5:
      predicted_sent = 1
    elif predicted_sent < 0:
      predicted_sent = -1
    else:
      predicted_sent = 0
    
    if predicted_stance >= 0.5:
      predicted_stance = 1
    elif predicted_sent < 0:
      predicted_stance = -1
    else:
      predicted_stance = 0

    return [predicted_sent, predicted_stance]

def accuracy(train_batch_acc, sent_nb_batches, stance_nb_batches):

  pred_sent = []
  pred_stance = []

  l = len(train_batch_acc)

  for i in range(l):
    pred_sent.append(train_batch_acc[i][0])
    pred_stance.append(train_batch_acc[i][1])

  print(pred_sent)
  print(sent_nb_batches)
  print(pred_stance)
  print(stance_nb_batches)

  sent_acc = accuracy_score(sent_nb_batches, pred_sent)
  stance_acc = accuracy_score(stance_nb_batches, pred_stance)
  
  return(sent_acc, stance_acc)


nb_epochs = 5
# batch_size = 47
batch_size = 1
nb_batches = 2914
# 2914

gen = batch_generator(batch_size, nb_batches)

model = JointMultiTaskModel()
adam = optim.Adam(model.parameters(), lr=learning_rate)

train_epoch_acc = []

for epoch in range(nb_epochs):

    train_batch_loss = []
    train_batch_acc = []

    sent_nb_batches = []
    stance_nb_batches = []

    for batch in range(nb_batches):
        # print("batch", batch)
        # print(len(next(gen)[0]))
        text, sent, stance = next(gen)
        # print(type(text))
        # print(text)
        # print(len(text))
        sent_nb_batches.append(sent[0][0])
        stance_nb_batches.append(stance[0][0])

        out = model.forward(text)

        # loss = model.loss(out, tags, chunks, sent)
        loss = model.loss(out, sent, stance)
        print("Epoch:", epoch,
              "Batch:", batch,
              "Loss:", loss.data[0])

        adam.zero_grad()
        # loss.backward()
        loss.sum().backward()
        adam.step()
        # print("out", out)
        model.eval() # enter evaluation mode
        with torch.no_grad():
              train_batch_acc.append(compare(out)) # evaluate mini-batch train accuracy in evaluation
    acc = accuracy(train_batch_acc, sent_nb_batches, stance_nb_batches)
    print("Epoch: ", epoch, "Sentiment Accuracy: ", acc[0], "Stance Accuracy: ", acc[1])

# train_epoch_acc.append(torch.tensor(train_batch_acc)
# prediction(model)
