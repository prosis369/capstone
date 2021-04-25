import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from utils import batch_generator
from utils import max_sentence_size
from utils import np2autograd
from sentiment import SentimentClassification
from stance import StanceClassification
from emotion import EmotionClassification
from bias import BiasClassification
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


dataset_file = './test.csv'

# Hyperparams
learning_rate = 1e-3
postag_reg = 1e-3
chunking_reg = 1e-3
sentiment_reg = 1e-3
stance_reg = 1e-3
emotion_reg = 1e-3
bias_reg = 1e-3

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

        self.emotion_anger = EmotionClassification()
        self.emotion_anticipation = EmotionClassification()
        self.emotion_disgust = EmotionClassification()
        self.emotion_fear = EmotionClassification()
        self.emotion_joy = EmotionClassification()
        self.emotion_sadness = EmotionClassification()
        self.emotion_surprise = EmotionClassification()
        self.emotion_trust = EmotionClassification()

        self.bias = BiasClassification()

        # Mode - all or module_name
        self.mode = mode

    def get_sentiment(self, sentence):
        return self.sentiment(sentence)

    def get_stance(self, sentence):
        return self.stance(sentence)
    
    def get_emotion_anger(self, sentence):
        return self.emotion_anger(sentence)

    def get_emotion_anticipation(self, sentence):
        return self.emotion_anticipation(sentence)
    
    def get_emotion_fear(self, sentence):
        return self.emotion_fear(sentence)
    
    def get_emotion_disgust(self, sentence):
        return self.emotion_disgust(sentence)
    
    def get_emotion_joy(self, sentence):
        return self.emotion_joy(sentence)

    def get_emotion_sadness(self, sentence):
        return self.emotion_sadness(sentence)
    
    def get_emotion_surprise(self, sentence):
        return self.emotion_surprise(sentence)

    def get_emotion_trust(self, sentence):
        return self.emotion_trust(sentence)

    def get_bias(self, sentence):
        return self.bias(sentence)

    def run_all(self, x):
        # sentence = self.embedded_batch(x)
        # print(x)
        for s in x:
            # print(len(s[0]))
            y_sentiment = self.get_sentiment(s)
            y_stance = self.get_stance(s)
            y_emotion_anger = self.get_emotion_anger(s)
            y_emotion_anticipation = self.get_emotion_anticipation(s)
            y_emotion_disgust = self.get_emotion_disgust(s)
            y_emotion_fear = self.get_emotion_fear(s)
            y_emotion_joy = self.get_emotion_joy(s)
            y_emotion_sadness = self.get_emotion_sadness(s)
            y_emotion_surprise = self.get_emotion_surprise(s)
            y_emotion_trust = self.get_emotion_trust(s)
            y_bias = self.get_bias(s)

            yield y_sentiment, y_stance, y_emotion_anger, y_emotion_anticipation, y_emotion_disgust, y_emotion_fear, y_emotion_joy, y_emotion_sadness, y_emotion_surprise, y_emotion_trust, y_bias

    def forward(self, x):
        # print(len(x))
        out = self.run_all(x)
        # print(out)
        out = list(out)

        return out


    def sentiment_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.sentiment.w.norm() ** 2) * sentiment_reg

        return loss

    def stance_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.stance.w.norm() ** 2) * stance_reg

        return loss
      
    def emotion_anger_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_anger.w.norm() ** 2) * emotion_reg

        return loss
    
    def emotion_anticipation_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_anticipation.w.norm() ** 2) * emotion_reg

        return loss
    
    def emotion_disgust_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_disgust.w.norm() ** 2) * emotion_reg

        return loss
    
    def emotion_fear_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_fear.w.norm() ** 2) * emotion_reg

        return loss

    def emotion_joy_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_joy.w.norm() ** 2) * emotion_reg

        return loss
    
    def emotion_sadness_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_sadness.w.norm() ** 2) * emotion_reg

        return loss
      
    def emotion_surprise_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_surprise.w.norm() ** 2) * emotion_reg

        return loss
    
    def emotion_trust_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.emotion_trust.w.norm() ** 2) * emotion_reg

        return loss

    def bias_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
               + (self.bias.w.norm() ** 2) * bias_reg

        return loss
      
    def loss(self, y, sentiment, stance, anger, anticipation, disgust, fear, joy, sadness, suprise, trust, bias):
        # print("Hi i am in loss")
        losses = []
        # print("Hi i am in loss line 2")
        length = len(y)

        for i in range(length):

            p_sent, r_sent = y[i][0], np2autograd(sentiment[i])
            p_stance, r_stance = y[i][1], np2autograd(stance[i])
            p_anger, r_anger = y[i][2], np2autograd(anger[i])
            p_anticipation, r_anticipation = y[i][3], np2autograd(anticipation[i])
            p_disgust, r_disgust = y[i][4], np2autograd(disgust[i])
            p_fear, r_fear = y[i][5], np2autograd(fear[i])
            p_joy, r_joy = y[i][6], np2autograd(joy[i])
            p_sadness, r_sadness = y[i][7], np2autograd(sadness[i])
            p_surprise, r_surprise = y[i][8], np2autograd(surprise[i])
            p_trust, r_trust = y[i][9], np2autograd(trust[i])
            p_bias, r_bias = y[i][10], np2autograd(bias[i])

            loss_sent = self.sentiment_loss(p_sent, r_sent)
            loss_stance = self.stance_loss(p_stance, r_stance)
            loss_emotion_anger = self.emotion_anger_loss(p_anger, r_anger)
            loss_emotion_anticipation = self.emotion_anticipation_loss(p_anticipation, r_anticipation)
            loss_emotion_disgust = self.emotion_disgust_loss(p_disgust, r_disgust)
            loss_emotion_fear = self.emotion_disgust_loss(p_fear, r_fear)
            loss_emotion_joy = self.emotion_disgust_loss(p_joy, r_joy)
            loss_emotion_sadness = self.emotion_disgust_loss(p_sadness, r_sadness)
            loss_emotion_surprise = self.emotion_disgust_loss(p_surprise, r_surprise)
            loss_emotion_trust = self.emotion_disgust_loss(p_trust, r_trust)
            loss_bias = self.bias_loss(p_bias, r_bias)

            sent_weight = 1
            stance_weight = 1
            bias_weight = 1
            emotion_weight = 1

            loss = (loss_sent * (1/4)) + (loss_stance * (1/4)) + (loss_emotion_anger * (1/32)) + (loss_emotion_anticipation * (1/32)) + (loss_emotion_disgust * (1/32)) + (loss_emotion_fear * (1/32)) + (loss_emotion_joy * (1/32)) + (loss_emotion_sadness * (1/32)) + (loss_emotion_surprise * (1/32)) + (loss_emotion_trust * (1/32)) + (loss_bias * (1/4))
            loss = loss/4

            losses.append(loss)

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

    # print(out)
    predicted_sent = out[0][0].numpy()[0][0]
    # print(predicted_sent)
    predicted_sent = threshold(predicted_sent, 0.5, 0)
    # print(predicted_sent)

    predicted_stance = out[0][1].numpy()[0][0]
    predicted_stance = threshold(predicted_stance, 0.5, 0)

    predicted_emotion_anger = out[0][2].numpy()[0][0]
    predicted_emotion_anger = threshold(predicted_emotion_anger, 0.5, 0)

    predicted_emotion_anticipation = out[0][3].numpy()[0][0]
    predicted_emotion_anticipation = threshold(predicted_emotion_anticipation, 0.5, 0)

    predicted_emotion_disgust = out[0][4].numpy()[0][0]
    predicted_emotion_disgust = threshold(predicted_emotion_disgust, 0.5, 0)

    predicted_emotion_fear = out[0][5].numpy()[0][0]
    predicted_emotion_fear = threshold(predicted_emotion_fear, 0.5, 0)

    predicted_emotion_joy = out[0][6].numpy()[0][0]
    predicted_emotion_joy = threshold(predicted_emotion_joy, 0.5, 0)

    predicted_emotion_sadness = out[0][7].numpy()[0][0]
    predicted_emotion_sadness = threshold(predicted_emotion_sadness, 0.5, 0)

    predicted_emotion_surprise = out[0][8].numpy()[0][0]
    predicted_emotion_surprise = threshold(predicted_emotion_surprise, 0.5, 0)

    predicted_emotion_trust = out[0][9].numpy()[0][0]
    predicted_emotion_trust = threshold(predicted_emotion_trust, 0.5, 0)

    predicted_bias = out[0][10].numpy()[0][0]
    predicted_bias = threshold(predicted_bias, 0.5, 0)
    
    return [predicted_sent, predicted_stance, predicted_emotion_anger, predicted_emotion_anticipation, predicted_emotion_disgust, predicted_emotion_fear, predicted_emotion_joy, predicted_emotion_sadness, predicted_emotion_surprise, predicted_emotion_trust, predicted_bias]

def accuracy(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches):

  pred_sent = []
  pred_stance = []
  pred_emotion_anger = []
  pred_emotion_anticipation = []
  pred_emotion_disgust = []
  pred_emotion_fear = []
  pred_emotion_joy = []
  pred_emotion_sadness = []
  pred_emotion_surprise = []
  pred_emotion_trust = []
  pred_bias = []

  l = len(train_batch_acc)

  for i in range(l):
    pred_sent.append(train_batch_acc[i][0])
    pred_stance.append(train_batch_acc[i][1])
    pred_emotion_anger.append(train_batch_acc[i][2])
    pred_emotion_anticipation.append(train_batch_acc[i][3])
    pred_emotion_disgust.append(train_batch_acc[i][4])
    pred_emotion_fear.append(train_batch_acc[i][5])
    pred_emotion_joy.append(train_batch_acc[i][6])
    pred_emotion_sadness.append(train_batch_acc[i][7])
    pred_emotion_surprise.append(train_batch_acc[i][8])
    pred_emotion_trust.append(train_batch_acc[i][9])
    pred_bias.append(train_batch_acc[i][10])

  print(pred_sent)
  print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)

  sent_acc = accuracy_score(sent_nb_batches, pred_sent)
  stance_acc = accuracy_score(stance_nb_batches, pred_stance)
  anger_acc = accuracy_score(emotion_anger_nb_batches, pred_emotion_anger)
  anticipation_acc = accuracy_score(emotion_anticipation_nb_batches, pred_emotion_anticipation)
  disgust_acc = accuracy_score(emotion_disgust_nb_batches, pred_emotion_disgust)
  fear_acc = accuracy_score(emotion_fear_nb_batches, pred_emotion_fear)
  joy_acc = accuracy_score(emotion_joy_nb_batches, pred_emotion_joy)
  sadness_acc = accuracy_score(emotion_sadness_nb_batches, pred_emotion_sadness)
  surprise_acc = accuracy_score(emotion_surprise_nb_batches, pred_emotion_surprise)
  trust_acc = accuracy_score(emotion_trust_nb_batches, pred_emotion_trust)
  bias_acc = accuracy_score(bias_nb_batches, pred_bias)
  
  return(sent_acc, stance_acc, anger_acc, anticipation_acc, disgust_acc, fear_acc, joy_acc, sadness_acc, surprise_acc, trust_acc, bias_acc)


def precision(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches):

  pred_sent = []
  pred_stance = []
  pred_emotion_anger = []
  pred_emotion_anticipation = []
  pred_emotion_disgust = []
  pred_emotion_fear = []
  pred_emotion_joy = []
  pred_emotion_sadness = []
  pred_emotion_surprise = []
  pred_emotion_trust = []
  pred_bias = []

  l = len(train_batch_acc)

  for i in range(l):
    pred_sent.append(train_batch_acc[i][0])
    pred_stance.append(train_batch_acc[i][1])
    pred_emotion_anger.append(train_batch_acc[i][2])
    pred_emotion_anticipation.append(train_batch_acc[i][3])
    pred_emotion_disgust.append(train_batch_acc[i][4])
    pred_emotion_fear.append(train_batch_acc[i][5])
    pred_emotion_joy.append(train_batch_acc[i][6])
    pred_emotion_sadness.append(train_batch_acc[i][7])
    pred_emotion_surprise.append(train_batch_acc[i][8])
    pred_emotion_trust.append(train_batch_acc[i][9])
    pred_bias.append(train_batch_acc[i][10])

  print(pred_sent)
  print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)

  sent_acc = precision_score(sent_nb_batches, pred_sent)
  stance_acc = precision_score(stance_nb_batches, pred_stance)
  anger_acc = precision_score(emotion_anger_nb_batches, pred_emotion_anger)
  anticipation_acc = precision_score(emotion_anticipation_nb_batches, pred_emotion_anticipation)
  disgust_acc = precision_score(emotion_disgust_nb_batches, pred_emotion_disgust)
  fear_acc = precision_score(emotion_fear_nb_batches, pred_emotion_fear)
  joy_acc = precision_score(emotion_joy_nb_batches, pred_emotion_joy)
  sadness_acc = precision_score(emotion_sadness_nb_batches, pred_emotion_sadness)
  surprise_acc = precision_score(emotion_surprise_nb_batches, pred_emotion_surprise)
  trust_acc = precision_score(emotion_trust_nb_batches, pred_emotion_trust)
  bias_acc = precision_score(bias_nb_batches, pred_bias)
  
  return(sent_acc, stance_acc, anger_acc, anticipation_acc, disgust_acc, fear_acc, joy_acc, sadness_acc, surprise_acc, trust_acc, bias_acc)


def recall(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches):

  pred_sent = []
  pred_stance = []
  pred_emotion_anger = []
  pred_emotion_anticipation = []
  pred_emotion_disgust = []
  pred_emotion_fear = []
  pred_emotion_joy = []
  pred_emotion_sadness = []
  pred_emotion_surprise = []
  pred_emotion_trust = []
  pred_bias = []

  l = len(train_batch_acc)

  for i in range(l):
    pred_sent.append(train_batch_acc[i][0])
    pred_stance.append(train_batch_acc[i][1])
    pred_emotion_anger.append(train_batch_acc[i][2])
    pred_emotion_anticipation.append(train_batch_acc[i][3])
    pred_emotion_disgust.append(train_batch_acc[i][4])
    pred_emotion_fear.append(train_batch_acc[i][5])
    pred_emotion_joy.append(train_batch_acc[i][6])
    pred_emotion_sadness.append(train_batch_acc[i][7])
    pred_emotion_surprise.append(train_batch_acc[i][8])
    pred_emotion_trust.append(train_batch_acc[i][9])
    pred_bias.append(train_batch_acc[i][10])

  print(pred_sent)
  print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)

  sent_acc = recall_score(sent_nb_batches, pred_sent)
  stance_acc = recall_score(stance_nb_batches, pred_stance)
  anger_acc = recall_score(emotion_anger_nb_batches, pred_emotion_anger)
  anticipation_acc = recall_score(emotion_anticipation_nb_batches, pred_emotion_anticipation)
  disgust_acc = recall_score(emotion_disgust_nb_batches, pred_emotion_disgust)
  fear_acc = recall_score(emotion_fear_nb_batches, pred_emotion_fear)
  joy_acc = recall_score(emotion_joy_nb_batches, pred_emotion_joy)
  sadness_acc = recall_score(emotion_sadness_nb_batches, pred_emotion_sadness)
  surprise_acc = recall_score(emotion_surprise_nb_batches, pred_emotion_surprise)
  trust_acc = recall_score(emotion_trust_nb_batches, pred_emotion_trust)
  bias_acc = recall_score(bias_nb_batches, pred_bias)
  
  return(sent_acc, stance_acc, anger_acc, anticipation_acc, disgust_acc, fear_acc, joy_acc, sadness_acc, surprise_acc, trust_acc, bias_acc)


def fscore(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches):

  pred_sent = []
  pred_stance = []
  pred_emotion_anger = []
  pred_emotion_anticipation = []
  pred_emotion_disgust = []
  pred_emotion_fear = []
  pred_emotion_joy = []
  pred_emotion_sadness = []
  pred_emotion_surprise = []
  pred_emotion_trust = []
  pred_bias = []

  l = len(train_batch_acc)

  for i in range(l):
    pred_sent.append(train_batch_acc[i][0])
    pred_stance.append(train_batch_acc[i][1])
    pred_emotion_anger.append(train_batch_acc[i][2])
    pred_emotion_anticipation.append(train_batch_acc[i][3])
    pred_emotion_disgust.append(train_batch_acc[i][4])
    pred_emotion_fear.append(train_batch_acc[i][5])
    pred_emotion_joy.append(train_batch_acc[i][6])
    pred_emotion_sadness.append(train_batch_acc[i][7])
    pred_emotion_surprise.append(train_batch_acc[i][8])
    pred_emotion_trust.append(train_batch_acc[i][9])
    pred_bias.append(train_batch_acc[i][10])

  print(pred_sent)
  print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)

  sent_acc = f1_score(sent_nb_batches, pred_sent)
  stance_acc = f1_score(stance_nb_batches, pred_stance)
  anger_acc = f1_score(emotion_anger_nb_batches, pred_emotion_anger)
  anticipation_acc = f1_score(emotion_anticipation_nb_batches, pred_emotion_anticipation)
  disgust_acc = f1_score(emotion_disgust_nb_batches, pred_emotion_disgust)
  fear_acc = f1_score(emotion_fear_nb_batches, pred_emotion_fear)
  joy_acc = f1_score(emotion_joy_nb_batches, pred_emotion_joy)
  sadness_acc = f1_score(emotion_sadness_nb_batches, pred_emotion_sadness)
  surprise_acc = f1_score(emotion_surprise_nb_batches, pred_emotion_surprise)
  trust_acc = f1_score(emotion_trust_nb_batches, pred_emotion_trust)
  bias_acc = f1_score(bias_nb_batches, pred_bias)
  
  return(sent_acc, stance_acc, anger_acc, anticipation_acc, disgust_acc, fear_acc, joy_acc, sadness_acc, surprise_acc, trust_acc, bias_acc)



nb_epochs = 1
# batch_size = 47
batch_size = 1
nb_batches = 50
# nb_batches = 1
# 2914
# 1956

gen = batch_generator(batch_size, nb_batches, dataset_file)

model = JointMultiTaskModel()
adam = optim.Adam(model.parameters(), lr=learning_rate)

train_epoch_acc = []

PATH = "saved_model.pt"

for epoch in range(nb_epochs):

    train_batch_loss = []
    train_batch_acc = []

    sent_nb_batches = []
    stance_nb_batches = []
    emotion_anger_nb_batches = []
    emotion_anticipation_nb_batches = []
    emotion_disgust_nb_batches = []
    emotion_fear_nb_batches = []
    emotion_joy_nb_batches = []
    emotion_sadness_nb_batches = []
    emotion_surprise_nb_batches = []
    emotion_trust_nb_batches = []
    bias_nb_batches = []

    for batch in range(nb_batches):

        text, sent, stance, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, bias = next(gen)

        sent_nb_batches.append(sent[0][0])
        stance_nb_batches.append(stance[0][0])
        emotion_anger_nb_batches.append(anger[0][0])
        emotion_anticipation_nb_batches.append(anticipation[0][0])
        emotion_disgust_nb_batches.append(disgust[0][0])
        emotion_fear_nb_batches.append(fear[0][0])
        emotion_joy_nb_batches.append(joy[0][0])
        emotion_sadness_nb_batches.append(sadness[0][0])
        emotion_surprise_nb_batches.append(surprise[0][0])
        emotion_trust_nb_batches.append(trust[0][0])
        bias_nb_batches.append(bias[0][0])
        
        out = model.forward(text)

        loss = model.loss(out, sent, stance, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, bias)
        print("Epoch:", epoch,
              "Batch:", batch,
              "Loss:", loss.data[0])

        adam.zero_grad()
        # loss.backward()
        loss.sum().backward()
        adam.step()
        # print("out", out)

        torch.save(model.state_dict(), PATH)


        model.eval() # enter evaluation mode
        with torch.no_grad():
              train_batch_acc.append(compare(out)) # evaluate mini-batch train accuracy in evaluation

    acc = accuracy(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches)
    prec = precision(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches)
    rec = recall(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches)
    fsc = fscore(train_batch_acc, sent_nb_batches, stance_nb_batches, emotion_anger_nb_batches, emotion_anticipation_nb_batches, emotion_disgust_nb_batches, emotion_fear_nb_batches, emotion_joy_nb_batches, emotion_sadness_nb_batches, emotion_surprise_nb_batches, emotion_trust_nb_batches, bias_nb_batches)

    print("Epoch: ", epoch, "Sentiment Accuracy: ", acc[0], "Stance Accuracy: ", acc[1], "Anger Accuracy: ", acc[2], "Anticipation Accuracy: ", acc[3], "Disgust Accuracy: ", acc[4], "Fear Accuracy: ", acc[5], "Joy Accuracy: ", acc[6], "Sadness Accuracy: ", acc[7], "Suprise Accuracy: ", acc[8], "Trust Accuracy: ", acc[9], "Bias Accuracy: ", acc[10])
    print("Epoch: ", epoch, "Sentiment Precision: ", prec[0], "Stance Precision: ", prec[1], "Anger Precision: ", prec[2], "Anticipation Precision: ", prec[3], "Disgust Precision: ", prec[4], "Fear Precision: ", prec[5], "Joy Precision: ", prec[6], "Sadness Precision: ", prec[7], "Suprise Precision: ", prec[8], "Trust Precision: ", prec[9], "Bias Precision: ", prec[10])
    print("Epoch: ", epoch, "Sentiment Recall: ", rec[0], "Stance Recall: ", rec[1], "Anger Recall: ", rec[2], "Anticipation Recall: ", rec[3], "Disgust Recall: ", rec[4], "Fear Recall: ", rec[5], "Joy Recall: ", rec[6], "Sadness Recall: ", rec[7], "Suprise Recall: ", rec[8], "Trust Recall: ", rec[9], "Bias Recall: ", rec[10])
    print("Epoch: ", epoch, "Sentiment F1-Score: ", fsc[0], "Stance F1-Score: ", fsc[1], "Anger F1-Score: ", fsc[2], "Anticipation F1-Score: ", fsc[3], "Disgust F1-Score: ", fsc[4], "Fear F1-Score: ", fsc[5], "Joy F1-Score: ", fsc[6], "Sadness F1-Score: ", fsc[7], "Suprise F1-Score: ", fsc[8], "Trust F1-Score: ", fsc[9], "Bias F1-Score: ", fsc[10])


