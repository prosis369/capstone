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
from transfer_bias import TransferBiasClassification

postag_reg = 1e-3
chunking_reg = 1e-3
sentiment_reg = 1e-3
stance_reg = 1e-3
emotion_reg = 1e-3
bias_reg = 1e-3

BIAS_PATH = "epoch_0saved_model_bias_toxic10kAndOlidFull.pt"

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

        self.bias = TransferBiasClassification()
        self.bias.load_state_dict(torch.load(BIAS_PATH))
        self.bias.eval()

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
      
    def loss(self, y, sentiment, stance, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, bias):
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

            loss = (loss_sent * (1/6)) + (loss_stance * (1/3)) + (loss_emotion_anger * (1/48)) + (loss_emotion_anticipation * (1/48)) + (loss_emotion_disgust * (1/48)) + (loss_emotion_fear * (1/48)) + (loss_emotion_joy * (1/48)) + (loss_emotion_sadness * (1/48)) + (loss_emotion_surprise * (1/48)) + (loss_emotion_trust * (1/48)) + (loss_bias * (1/3))
            loss = loss/4

            losses.append(loss)

        loss = losses[0]

        for i in range(1, length):
            loss += losses[i]

        loss = loss / length

        return loss

