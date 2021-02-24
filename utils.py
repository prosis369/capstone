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


# The chunk gram regex for the chunking parser
# chunk_gram = r"""
# NP: {<DT|JJ|NN.*>+}             # Chunk sequences of DT, JJ, NN
# PP: {<IN><NP>}                  # Chunk prepositions followed by NP
# VP: {<VB.*><NP|PP|CLAUSE>+$}    # Chunk verbs and their arguments
# ACTION: {<PRP><VBP><VBN>}
# CLAUSE: {<NP><VP>}              # Chunk NP, VP
# SENT: {<.*>+}
#       }<.>{
# """
#
# chunk_tags = ['NP', 'PP', 'VP', 'ACTION', 'CLAUSE', 'SENT', 'S']
#
# nb_chunktags = len(chunk_tags)
#
# chk2k = dict([(v, k) for k, v in enumerate(chunk_tags)])
# k2chk = dict([(k, v) for k, v in enumerate(chunk_tags)])
#
# chunk_parser = nltk.RegexpParser(chunk_gram)
#
# # Postags to consider
# postags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
#            "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP",
#            "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG",
#            "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "UKN"]
#
# nb_postags = len(postags)
#
# t2k = dict([(v, k) for k, v in enumerate(postags)])
# k2t = dict([(k, v) for k, v in enumerate(postags)])


def get_dataset(batch_size, skip=None):
    """ Gets the dataset iterator
    """
    # dataset = pd.read_csv(dataset_file,
    #                       iterator=True,
    #                       skiprows=range(1, skip * batch_size) if skip else None,
    #                       chunksize=batch_size)

    skiprowsValue = 0
    if skip:
      skiprowsValue = skip*batch_size
    else:
      skiprowsValue = None
    dataset = pd.read_csv(dataset_file,
                          iterator=True,
                          skiprows = [i for i in range(1, skiprowsValue)] if skip else None,
                          chunksize=batch_size)

    return dataset

def batch_generator(batch_size, nb_batches, skip_batches=None):
    """ Batch generator for the many task joint model.
    """
    batch_count = 0
    dataset = get_dataset(batch_size, skip_batches)
    batch_number = 1

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

        yield text, sent, stance

        batch_count += 1

        if batch_count >= nb_batches:
            dataset = get_dataset(batch_size, batch_number*nb_batches)
            batch_number += 1
            batch_count = 0

# Bag of Chars
# boc = get_dataset(boc_size).get_chunk()
# boc = boc['Text'].values
# boc = "".join(boc).lower()
# boc = set(boc)
# boc = sorted(boc)
#
# # Character Classes Dictionaries
# c2k = dict([(v, k) for k, v in enumerate(boc)])
# k2c = dict([(k, v) for k, v in enumerate(boc)])
#
# nb_classes = len(boc)
#
# del boc


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
  

# def vec2sent(vec):
#     """ Converts a char-vector word representation
#         sentence to its respective string.
#     """
#     words = [vec2word(v) for v in vec if type(v) == list]
#     sent = " ".join(words)

#     return sent


# def sent2tags(sentence):
#     """ Returns a vector of tag-classes from a given
#         sentence.
#     """
#     tags = word_tokenize(sentence)
#     tags = nltk.pos_tag(tags)
#     out = []
#
#     for _, tag in tags[:max_sentence_size]:
#         if tag in postags:
#             out.append(t2k.get(tag))
#         else:
#             out.append(t2k.get("UKN"))
#
#     return out


# def tags2sent(tags):
#     """ Returns the string representation of a given
#         vector of tag-classes.
#     """
#     sent = map(k2t.get, tags)
#     sent = " ".join(tags)
#
#     return sent


# def sent2chunk(sentence):
#     """ Returns the chunking classes of a given
#         sentence.
#
#         Given:
#         I like this product, but I wouldn't recommend it.
#
#         The chunk tree is formed such as:
#         S                   S                             S
#                             ,                             .
#         SENT SENT             SENT SENT SENT SENT SENT
#         I like
#                NP NP
#                this product
#                               but I wouldn't recomment it
#
#         Then, the resulted vector would be:
#         [SENT, SENT, NP, NP, S, SENT, SENT, SENT, SENT, SENT, S]
#     """
#     # Get pos tags
#     tags = word_tokenize(sentence)
#     tags = nltk.pos_tag(tags)
#     tags = tags[:max_sentence_size]
#
#     # Chunks it
#     chunked = chunk_parser.parse(tags)
#
#     flatten = [chunk for chunk in chunked]
#     out = ['S'] * len(tags)
#     pointer = 0
#
#     # Parses all sublevel tree in order to flatten it
#     # to its respective chunk-tags.
#     while pointer < len(out):
#         item = flatten[pointer]
#
#         if type(item) == tuple:
#             if type(item[1]) == nltk.tree.Tree:
#                 item = item[1]
#             else:
#                 if item[0] in chunk_tags:
#                     out[pointer] = item[0]
#                 pointer += 1
#                 continue
#
#         flatten = flatten[:pointer] + flatten[pointer + 1:]
#         for i, leaf in enumerate(item):
#             flatten = flatten[:pointer + i] + [(item._label, leaf)] + flatten[pointer + i:]
#
#     out = [chk2k.get(chk) for chk in out]
#
#     return out