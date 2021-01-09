# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

import numpy as np
from gensim.models import Word2Vec, Phrases
from tqdm import tqdm
import cPickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
vocabulary = cPickle.load(open('vocabularies/vocabulary.pickle', 'rb'))

dummy_word_idx = vocabulary.get('DUMMY_WORD', None)
print("alphabet", len(vocabulary))
print('dummy_word:', dummy_word_idx)

filtered_vocab = {}

new_idx = 0
for word, (idx,freq) in tqdm(vocabulary.items()):
    if freq > 45:
        filtered_vocab[word] = (new_idx, freq)
        new_idx += 1

filtered_vocab['UNK'] = (new_idx + 1, 100)
filtered_vocab['DUMMY_WORD'] = (new_idx + 2, 100)
print("alphabet", len(filtered_vocab))

model = Word2Vec.load_word2vec_format('vocabularies/embedding_file')
#create numpy embedding matrix as input for the EmbeddingLayer in Keras
vocab_emb = np.zeros((len(filtered_vocab) + 1, 200), dtype='float32')
for word, (idx, freq) in tqdm(filtered_vocab.items()):
    word_vec = None
    if model.vocab.get(word, None):
        word_vec = model[word]
    if word_vec == None:
        word_vec = np.random.uniform(-0.25, 0.25, 200)
    vocab_emb[idx] = word_vec

outfile = os.path.join('vocabularies', 'embedding_matrix.npy')
np.save(outfile, vocab_emb)
cPickle.dump(filtered_vocab, open(os.path.join('vocabularies', 'vocabulary_filtered.pickle'), 'wb'))
