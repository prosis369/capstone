import nltk
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from transformers import BertTokenizer
from transformers import AutoTokenizer

dataset_file = './train.csv'

from pytorch_pretrained_bert import BertTokenizer, BertConfig
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



max_sentence_size = 300
max_word_size = 15

# The number of lines to read from the dataset to build
# key value dicts
boc_size = 10000


def np2autograd(var):
    var = np.array(var, dtype=np.float32)
    var = torch.from_numpy(var)
    var = Variable(var).long()

    return var


def avg_cross_entropy_loss(predicted, targets):
    """ Helper function for computing the simple mean
        cross entropy loss between the predicted one-hot
        and the target class.
    """
    losses = []
    length = len(predicted)

    for i in range(length):
        target = np.array(targets[i], dtype=np.float32)
        target = torch.from_numpy(target)
        target = Variable(target).long()

        loss = F.cross_entropy(predicted[i], target)

        losses.append(loss)

    loss = losses[0]

    for i in range(1, length):
        loss += losses[i]

    loss = loss / length

    return loss


def get_dataset(batch_size, skip=None):
    """ Gets the dataset iterator
    """
    # dataset = pd.read_csv(dataset_file,
    #                       iterator=True,
    #                       skiprows=range(1, skip * batch_size) if skip else None,
    #                       chunksize=batch_size)

    # skiprowsValue = 0
    # if skip:
    #   skiprowsValue = skip*batch_size
    # else:
    #   skiprowsValue = None
    dataset = pd.read_csv(dataset_file,
                          iterator=True,
                          # skiprows = [i for i in range(1, skiprowsValue)] if skip else None,
                          skiprows=range(1, skip * batch_size) if skip else None,
                          chunksize=batch_size)

    return dataset

def batch_generator(batch_size, nb_batches, skip_batches=None):
    """ Batch generator for the many task joint model.
    """
    batch_count = 0
    dataset = get_dataset(batch_size, skip_batches)
    # batch_number = 1

    while True:
        chunk = dataset.get_chunk()

        # text, tags, chunks = [], [], []
        text = []

        # print(len(chunk['Tweet'].values))
        

        for sent in chunk['Tweet'].values:
            # print(len(sent))
            # tags.append(sent2tags(sent))
            # sent = sent[:4]
            # text.append(sent2vec(sent))
            text.append(sent2bert(sent))
            # chunks.append(sent2chunk(sent))

        # The sentiment of the review where 1 is positive and 0 is negative
        # sent = (chunk['Score'] >= 4).values
        # sent = np.int32(sent).reshape(-1, 1)

        sent = (chunk['Sentiment']).values
        sent = np.int32(sent).reshape(-1, 1)
        stance = (chunk['Stance']).values
        stance = np.int32(stance).reshape(-1, 1)
        anger = (chunk['Anger']).values
        anger = np.int32(sent).reshape(-1, 1)
        anticipation = (chunk['Anticipation']).values
        anticipation = np.int32(sent).reshape(-1, 1)
        disgust = (chunk['Disgust']).values
        disgust = np.int32(sent).reshape(-1, 1)
        fear = (chunk['Fear']).values
        fear = np.int32(sent).reshape(-1, 1)
        joy = (chunk['Joy']).values
        joy = np.int32(sent).reshape(-1, 1)
        sadness = (chunk['Sadness']).values
        sadness = np.int32(sent).reshape(-1, 1)
        surprise = (chunk['Surprise']).values
        surprise = np.int32(sent).reshape(-1, 1)
        trust = (chunk['Trust']).values
        trust = np.int32(sent).reshape(-1, 1)

        yield text, sent, stance, anger, anticipation, disgust, fear, joy, sadness, surprise, trust 

        batch_count += 1

        if batch_count >= nb_batches:
            # dataset = get_dataset(batch_size, batch_number*nb_batches)
            # batch_number += 1
            dataset = get_dataset(batch_size)
            batch_count = 0


# def word2vec(word):
#     """ Converts a word to its char-vector.
#     """
#     vec = map(c2k.get, word.lower())
#     vec = list(vec)
#
#     return vec
#
#
# def vec2word(vec):
#     """ Converts a char-vector to its respective
#         string.
#     """
#     word = map(k2c.get, vec)
#     word = "".join(word)
#
#     return word


def sent2vec(sentence):
    """ Returns the char-vector word representation
        of a given sentence.
    """
    # print(sentence)
    tokens = [tokenizer.tokenize(sentence)]
    # print((tokens))
    # tokens = tokens.numpy
    vecs = [tokenizer.convert_tokens_to_ids(tokens[0])]
    # print(len(vecs[0]))
    return vecs
'''
def preprocessing_for_bert(data, tokenizer):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            truncation = True,
            return_attention_mask=True      # Return attention mask
            )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks
'''
def sent2bert(sentence):
  '''
    MAX_SEQUENCE_LENGTH = 100
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # tokenized_texts = [tokenizer.tokenize(sent) for sent in df['Tweet'].values]
    tokenized_texts = [tokenizer.tokenize(sentence)]
    # tokenizer.convert_tokens_to_ids(tokenizer.tokenize())
    # vecs = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
    vecs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_SEQUENCE_LENGTH, dtype="long", truncating="post", padding="post")
    print(len(vecs[0]))
    return vecs

  
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  token_ids = list(preprocessing_for_bert([sentence], tokenizer)[0].squeeze().numpy())
  return token_ids
  '''
  # print(len(sentence))
  tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
  batch = tokenizer(sentence, max_length = 512, padding='max_length', truncation=True, return_tensors="pt")
  # print(batch['input_ids'].tolist())
  return batch['input_ids'].tolist()
  