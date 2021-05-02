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

# print(model)


def threshold_sentiment(prediction, upperBound, lowerBound):
  # train_predictions_sent.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  elif prediction < lowerBound:
      prediction = -1
  else:
      prediction = 0
  return prediction

def threshold_stance(prediction, upperBound, lowerBound):
  # train_predictions_stance.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  elif prediction < lowerBound:
      prediction = -1
  else:
      prediction = 0
  return prediction

def threshold_emotion_anger(prediction, upperBound, lowerBound):
  # train_predictions_anger.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_anticipation(prediction, upperBound, lowerBound):
  # train_predictions_anticipation.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_disgust(prediction, upperBound, lowerBound):
  # train_predictions_disgust.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_fear(prediction, upperBound, lowerBound):
  # train_predictions_fear.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_joy(prediction, upperBound, lowerBound):
  # train_predictions_joy.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_sadness(prediction, upperBound, lowerBound):
  # train_predictions_sadness.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_surprise(prediction, upperBound, lowerBound):
  # train_predictions_surprise.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_emotion_trust(prediction, upperBound, lowerBound):
  # train_predictions_trust.append(prediction)

  if prediction >= upperBound:
      prediction = 1
  else:
      prediction = 0
  return prediction

def threshold_bias(prediction, upperBound, lowerBound):
  # train_predictions_bias.append(prediction)

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
    predicted_emotion_anger = threshold_emotion_anger(predicted_emotion_anger, 0.5, 0)

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

  # print(pred_sent)
  # print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)
  # print(emotion_anger_nb_batches)
  # print(pred_emotion_anger)

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

  # print(pred_sent)
  # print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)

  sent_acc = precision_score(sent_nb_batches, pred_sent, average="weighted")
  stance_acc = precision_score(stance_nb_batches, pred_stance,average="weighted")
  anger_acc = precision_score(emotion_anger_nb_batches, pred_emotion_anger,average="weighted")
  anticipation_acc = precision_score(emotion_anticipation_nb_batches, pred_emotion_anticipation,average="weighted")
  disgust_acc = precision_score(emotion_disgust_nb_batches, pred_emotion_disgust,average="weighted")
  fear_acc = precision_score(emotion_fear_nb_batches, pred_emotion_fear,average="weighted")
  joy_acc = precision_score(emotion_joy_nb_batches, pred_emotion_joy,average="weighted")
  sadness_acc = precision_score(emotion_sadness_nb_batches, pred_emotion_sadness,average="weighted")
  surprise_acc = precision_score(emotion_surprise_nb_batches, pred_emotion_surprise,average="weighted")
  trust_acc = precision_score(emotion_trust_nb_batches, pred_emotion_trust,average="weighted")
  bias_acc = precision_score(bias_nb_batches, pred_bias,average="weighted")
  
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

  # print(pred_sent)
  # print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)

  sent_acc = recall_score(sent_nb_batches, pred_sent,average="weighted")
  stance_acc = recall_score(stance_nb_batches, pred_stance,average="weighted")
  anger_acc = recall_score(emotion_anger_nb_batches, pred_emotion_anger,average="weighted")
  anticipation_acc = recall_score(emotion_anticipation_nb_batches, pred_emotion_anticipation,average="weighted")
  disgust_acc = recall_score(emotion_disgust_nb_batches, pred_emotion_disgust,average="weighted")
  fear_acc = recall_score(emotion_fear_nb_batches, pred_emotion_fear,average="weighted")
  joy_acc = recall_score(emotion_joy_nb_batches, pred_emotion_joy,average="weighted")
  sadness_acc = recall_score(emotion_sadness_nb_batches, pred_emotion_sadness,average="weighted")
  surprise_acc = recall_score(emotion_surprise_nb_batches, pred_emotion_surprise,average="weighted")
  trust_acc = recall_score(emotion_trust_nb_batches, pred_emotion_trust,average="weighted")
  bias_acc = recall_score(bias_nb_batches, pred_bias,average="weighted")
  
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

  # print(pred_sent)
  # print(sent_nb_batches)
  # print(pred_stance)
  # print(stance_nb_batches)

  sent_acc = f1_score(sent_nb_batches, pred_sent,average="weighted")
  stance_acc = f1_score(stance_nb_batches, pred_stance,average="weighted")
  anger_acc = f1_score(emotion_anger_nb_batches, pred_emotion_anger,average="weighted")
  anticipation_acc = f1_score(emotion_anticipation_nb_batches, pred_emotion_anticipation,average="weighted")
  disgust_acc = f1_score(emotion_disgust_nb_batches, pred_emotion_disgust,average="weighted")
  fear_acc = f1_score(emotion_fear_nb_batches, pred_emotion_fear,average="weighted")
  joy_acc = f1_score(emotion_joy_nb_batches, pred_emotion_joy,average="weighted")
  sadness_acc = f1_score(emotion_sadness_nb_batches, pred_emotion_sadness,average="weighted")
  surprise_acc = f1_score(emotion_surprise_nb_batches, pred_emotion_surprise,average="weighted")
  trust_acc = f1_score(emotion_trust_nb_batches, pred_emotion_trust,average="weighted")
  bias_acc = f1_score(bias_nb_batches, pred_bias,average="weighted")
  
  return(sent_acc, stance_acc, anger_acc, anticipation_acc, disgust_acc, fear_acc, joy_acc, sadness_acc, surprise_acc, trust_acc, bias_acc)

def prediction(model):
    dataset_file = './test-edit-final.csv'

    dataset = pd.read_csv(dataset_file)
    tweets = list(dataset['Tweet'])
    # print(tweets[0])
    # print(tweets[0])

    y_pred = []
    # for i in range(468,488):
    for i in range(len(tweets)):
        row = sent2bert(tweets[i])
        yhat = model.forward([row])
        # print(type(yhat))
        # yhat = yhat.detach().numpy()
        print("Tweet", i, "done in test dataset")
        y_pred.append(compare(yhat))
    # print(row)

    # test_acc = accuracy(y_pred, list(dataset['Sentiment'])[468:488],list(dataset['Stance'])[468:488])
    test_acc = accuracy(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))
    print("Test dataset: ", "Sentiment Accuracy: ", test_acc[0], "Stance Accuracy: ", test_acc[1], "Anger Accuracy: ", test_acc[2],"Anticipation Accuracy: ", test_acc[3],"Disgust Accuracy: ", test_acc[4],"Fear Accuracy: ", test_acc[5], "Joy Accuracy: ", test_acc[6], "Sadness Accuracy: ", test_acc[7], "Suprise Accuracy: ", test_acc[8], "Trust Accuracy: ", test_acc[9], "Bias Accuracy: ", test_acc[10])
    
    prec = precision(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))
    rec = recall(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))
    fsc = fscore(y_pred, list(dataset['Sentiment']),list(dataset['Stance']), list(dataset['Anger']), list(dataset['Anticipation']), list(dataset['Disgust']), list(dataset['Fear']), list(dataset['Joy']), list(dataset['Sadness']), list(dataset['Surprise']), list(dataset['Trust']), list(dataset['Bias']))

    print("Test dataset: ", "Sentiment Precision: ", prec[0], "Stance Precision: ", prec[1], "Anger Precision: ", prec[2], "Anticipation Precision: ", prec[3], "Disgust Precision: ", prec[4], "Fear Precision: ", prec[5], "Joy Precision: ", prec[6], "Sadness Precision: ", prec[7], "Suprise Precision: ", prec[8], "Trust Precision: ", prec[9], "Bias Precision: ", prec[10])
    print("Test dataset: ", "Sentiment Recall: ", rec[0], "Stance Recall: ", rec[1], "Anger Recall: ", rec[2], "Anticipation Recall: ", rec[3], "Disgust Recall: ", rec[4], "Fear Recall: ", rec[5], "Joy Recall: ", rec[6], "Sadness Recall: ", rec[7], "Suprise Recall: ", rec[8], "Trust Recall: ", rec[9], "Bias Recall: ", rec[10])
    print("Test dataset: ", "Sentiment F1-Score: ", fsc[0], "Stance F1-Score: ", fsc[1], "Anger F1-Score: ", fsc[2], "Anticipation F1-Score: ", fsc[3], "Disgust F1-Score: ", fsc[4], "Fear F1-Score: ", fsc[5], "Joy F1-Score: ", fsc[6], "Sadness F1-Score: ", fsc[7], "Suprise F1-Score: ", fsc[8], "Trust F1-Score: ", fsc[9], "Bias F1-Score: ", fsc[10])


    # print(yhat)

prediction(model)