import numpy as np
from nltk.tokenize import TweetTokenizer
import gzip
import re
import operator


def preprocess_tweet(tweet):
    #lowercase and normalize urls
    tweet = tweet.replace('\n', '').lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', tweet)
    tweet = re.sub('@[^\s]+', '<user>', tweet)
    #tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet

UNKNOWN_WORD_IDX = 0


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=140, verbose=0):
    UNKNOWN_WORD_IDX = alphabet.get('UNK', (0, 1))[0]
    data_idx = []
    max_len = 0
    unknown_words = 0
    known_words = 0
    for sentence in data:
        ex = np.ones(max_sent_length) * dummy_word_idx
        max_len = max(len(sentence),max_len)
        if len(sentence) > max_sent_length:
            sentence = sentence[:max_sent_length]
        for i, token in enumerate(sentence):
            idx, freq = alphabet.get(token, (0, 0))
            ex[i] = idx
            if idx == UNKNOWN_WORD_IDX:
                unknown_words += 1
            else:
                known_words += 1
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    if verbose == 1:
        print("Max length in this batch:",max_len)
        print("Number of unknown words:",unknown_words)
        print("Number of known words:",known_words)
    return data_idx

