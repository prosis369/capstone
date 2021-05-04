import torch
from utils import sent2bert
import pandas as pd

from torch.autograd import Variable
from joint_model import JointMultiTaskModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


PATH = "saved_model_final_1.pt"
model = JointMultiTaskModel()
model.load_state_dict(torch.load(PATH))
model.eval()


def threshold_sentiment(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  elif prediction < lowerBound:
      prediction = -1
  else:
      prediction = 0
  return prediction

def threshold_stance(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  elif prediction < lowerBound:
      prediction = -1
  else:
      prediction = 0
  return prediction

def threshold_emotion_anger(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_anticipation(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_disgust(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_fear(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_joy(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def cleanTweet(tweet):

  prediction = tweet[len(tweet)-1]
  global preb
  if prediction == '.':
    preb = 0
  else:
    preb = 1
  return prediction

def threshold_emotion_sadness(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_surprise(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_trust(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_bias(prediction, upperBound, lowerBound):

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction


def compare(out):

    # print(out)
    predicted_sent = out[0][0]
    # print(predicted_sent)
    predicted_sent = threshold_sentiment(predicted_sent, 0.5, 0)
    # print(predicted_sent)

    predicted_stance = out[0][1]
    predicted_stance = threshold_stance(predicted_stance, 0.5, 0)

    predicted_emotion_anger = out[0][2]
    predicted_emotion_anger = threshold_emotion_anticipation(predicted_emotion_anger, 0.5, 0)

    predicted_emotion_anticipation = out[0][3]
    predicted_emotion_anticipation = threshold_emotion_anticipation(predicted_emotion_anticipation, 0.5, 0)

    predicted_emotion_disgust = out[0][4]
    predicted_emotion_disgust = threshold_emotion_disgust(predicted_emotion_disgust, 0.5, 0)

    predicted_emotion_fear = out[0][5]
    predicted_emotion_fear = threshold_emotion_fear(predicted_emotion_fear, 0.5, 0)

    predicted_emotion_joy = out[0][6]
    predicted_emotion_joy = threshold_emotion_joy(predicted_emotion_joy, 0.5, 0)

    predicted_emotion_sadness = out[0][7]
    predicted_emotion_sadness = threshold_emotion_sadness(predicted_emotion_sadness, 0.5, 0)

    predicted_emotion_surprise = out[0][8]
    predicted_emotion_surprise = threshold_emotion_surprise(predicted_emotion_surprise, 0.5, 0)

    predicted_emotion_trust = out[0][9]
    predicted_emotion_trust = threshold_emotion_trust(predicted_emotion_trust, 0.5, 0)

    predicted_bias = out[0][10]
    predicted_bias = threshold_bias(predicted_bias, 0.5, 0)
    
    return [predicted_sent, predicted_stance, predicted_emotion_anger, predicted_emotion_anticipation, predicted_emotion_disgust, predicted_emotion_fear, predicted_emotion_joy, predicted_emotion_sadness, predicted_emotion_surprise, predicted_emotion_trust, predicted_bias]

def prediction_value_stance(pred):
  if pred == 0:
    return "Against"
  elif pred == 1:
    return "Favour"
  else:
    return "Neutral"

def prediction_value_sentiment(pred):
  if preb == 0:
    return "Negative"
  elif preb == 1:
    return "Positive"
  else:
    return "Neutral"

def prediction_value_emotion(pred):
  if preb == 0:
    return "Yes"
  else:
    return "No"

def prediction_value_emotiom(pred):
  if preb == 0:
    return "No"
  else:
    return "Yes"

def prediction_value_bias(pred):
  if preb == 0:
    return "Yes"
  else:
    return "No"

def prediction(model):
    
    '''
    0 - against
    1 - favour
    -1 - neutral

    0 - not anger
    1 - anger
    '''

    tweet = input("Enter the tweet: ")
    cleanTweet(tweet)
    row = sent2bert(tweet)
    yhat = model.forward([row])

    print("Tweet is processed")
    y_pred = compare(yhat)

    print("Sentiment: ", prediction_value_sentiment(y_pred[0]))
    print("Stance: ", prediction_value_stance(y_pred[1]))
    print("Anger: ", prediction_value_emotion(y_pred[2]))
    print("Anticipation: ", prediction_value_emotiom(y_pred[3]))
    print("Disgust: ", prediction_value_emotion(y_pred[4]))
    print("Fear: ", prediction_value_emotion(y_pred[5]))
    print("Joy: ", prediction_value_emotiom(y_pred[6]))
    print("Sadness: ", prediction_value_emotion(y_pred[7]))
    print("Surprise: ", prediction_value_emotiom(y_pred[8]))
    print("Trust: ", prediction_value_emotiom(y_pred[9]))
    print("Bias: ", prediction_value_bias(y_pred[10]))

    # print(row)

    # test_acc = accuracy(y_pred, list(dataset['Sentiment'])[468:488],list(dataset['Stance'])[468:488])
    # test_acc = accuracy(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))
    # print("Test dataset: ", "Sentiment Accuracy: ", test_acc[0], "Stance Accuracy: ", test_acc[1], "Anger Accuracy: ", test_acc[2],"Anticipation Accuracy: ", test_acc[3],"Disgust Accuracy: ", test_acc[4],"Fear Accuracy: ", test_acc[5], "Joy Accuracy: ", test_acc[6], "Sadness Accuracy: ", test_acc[7], "Suprise Accuracy: ", test_acc[8], "Trust Accuracy: ", test_acc[9], "Bias Accuracy: ", test_acc[10])
    
    # prec = precision(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))
    # rec = recall(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))
    # fsc = fscore(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))

    # print("Test dataset: ", "Sentiment Precision: ", prec[0], "Stance Precision: ", prec[1], "Anger Precision: ", prec[2], "Anticipation Precision: ", prec[3], "Disgust Precision: ", prec[4], "Fear Precision: ", prec[5], "Joy Precision: ", prec[6], "Sadness Precision: ", prec[7], "Suprise Precision: ", prec[8], "Trust Precision: ", prec[9], "Bias Precision: ", prec[10])
    # print("Test dataset: ", "Sentiment Recall: ", rec[0], "Stance Recall: ", rec[1], "Anger Recall: ", rec[2], "Anticipation Recall: ", rec[3], "Disgust Recall: ", rec[4], "Fear Recall: ", rec[5], "Joy Recall: ", rec[6], "Sadness Recall: ", rec[7], "Suprise Recall: ", rec[8], "Trust Recall: ", rec[9], "Bias Recall: ", rec[10])
    # print("Test dataset: ", "Sentiment F1-Score: ", fsc[0], "Stance F1-Score: ", fsc[1], "Anger F1-Score: ", fsc[2], "Anticipation F1-Score: ", fsc[3], "Disgust F1-Score: ", fsc[4], "Fear F1-Score: ", fsc[5], "Joy F1-Score: ", fsc[6], "Sadness F1-Score: ", fsc[7], "Suprise F1-Score: ", fsc[8], "Trust F1-Score: ", fsc[9], "Bias F1-Score: ", fsc[10])


    # print(yhat)
preb = 0
prediction(model)