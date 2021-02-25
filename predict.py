from utils import sent2bert
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from joint_model import model
# from joint_model import accuracy

def compare(out):

    # print(out)
    # actual_sent = actual_sent[0][0]
    # actual_stance = actual_stance[0][0]
    # print(out[0][0].detach())
    # print(out[0][1].detach())

    # print(out[0][0].numpy()[0][0])
    # predicted_sent = out[0][0].numpy()[0][0]
    # predicted_stance = out[0][1].numpy()[0][0]
    predicted_sent = out[0][0]
    predicted_stance = out[0][1]

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


def prediction(model):
    dataset_file = './test.csv'

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
    test_acc = accuracy(y_pred, list(dataset['Sentiment']),list(dataset['Stance']))
    print("Test dataset:", "Sentiment Accuracy: ", test_acc[0], "Stance Accuracy: ", test_acc[1])
    # print(yhat)

prediction(model)