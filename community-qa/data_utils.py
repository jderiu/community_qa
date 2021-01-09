import numpy as np
from parse_utils import preprocess_tweet, convert2indices
from collections import defaultdict
import os
import json
from alphabet import Alphabet
from nltk.tokenize import TweetTokenizer
from gensim.models import Phrases
import sys
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle

def convert_label(label):
    return {
        'Good': 1,
    }.get(label, 0)


def overlap_features(data1, data2, word2df):
    feats_overlap = []
    for d1, d2 in zip(data1, data2):
        d1_set = set(d1)
        d2_set = set(d2)
        word_overlap = d1_set.intersection(d2_set)
        overlap = float(len(word_overlap)) / (len(d1_set) + len(d2_set))
        df_overlap = 0.0
        for w in word_overlap:
            df_overlap += word2df[w]
        df_overlap /= (len(d1_set) + len(d2_set))
        feats_overlap.append(np.array([
            overlap,
            df_overlap
        ]))
    return np.array(feats_overlap)


def overlap_indices(data1, data2, max_len1, max_len2):
    data1_idx, data2_idx = [], []
    for d1, d2 in zip(data1, data2):
        d1_set = set(d1)
        d2_set = set(d2)
        overlap = d1_set.intersection(d2_set)

        d1_idx = np.ones(max_len1) * 2
        for idx, d1_word in enumerate(d1):
            if d1_word in overlap:
                d1_idx[idx] = 1
            else:
                d1_idx[idx] = 0
        data1_idx.append(d1_idx)

        d2_idx = np.ones(max_len2) * 2
        for idx, d2_word in enumerate(d2):
            if d2_word in overlap:
                d2_idx[idx] = 1
            else:
                d2_idx[idx] = 0
        data2_idx.append(d2_idx)

    d1_indices = np.vstack(data1_idx).astype('int32')
    d2_indices = np.vstack(data2_idx).astype('int32')

    return d1_indices, d2_indices


def compute_dfs(docs):
    word2df = defaultdict(float)
    for doc in docs:
        for w in set(doc):
            word2df[w] += 1.0
    n_docs = len(docs)
    for w, value in word2df.items():
        word2df[w] /= np.math.log(n_docs/value)
    return word2df


def get_unique_questions(q_ids, questions):
    question_for_id = {}
    for id, question in zip(q_ids, questions):
        question_for_id[id] = question
    return list(question_for_id.values())


def apply_transformer(sentence, transofrmers):
    for transofrmer in transofrmers:
        sentence = transofrmer[sentence]
    return sentence


def load_data_taskA(fname, alphabet, tknzr, transformers, max_len=None, text_label=4, ignore_label=False):
    dummy_word_idx = alphabet.get('DUMMY_WORD', (1, 1))[0]

    data_raw_loaded = open(fname, 'rt').readlines()
    data_raw = [x.split('\t') for x in data_raw_loaded]

    relq_id = np.asarray([x[0] for x in data_raw])
    relc_id = np.asarray([x[1] for x in data_raw])

    subject_tok = [tknzr.tokenize(preprocess_tweet(x[2])) for x in data_raw]
    question_tok = [tknzr.tokenize(preprocess_tweet(x[3])) for x in data_raw]
    comment_tok =[tknzr.tokenize(preprocess_tweet(x[text_label])) for x in data_raw]

    subject_tok = [apply_transformer(x, transformers) for x in subject_tok]
    question_tok = [apply_transformer(x, transformers) for x in question_tok]
    comment_tok = [apply_transformer(x, transformers) for x in comment_tok]

    if not max_len:
        subj_max_len = max([len(x) for x in subject_tok])
        ques_max_len = max([len(x) for x in question_tok])
        comm_max_len = max([len(x) for x in comment_tok])
        max_len = (subj_max_len, ques_max_len, comm_max_len, 2)
    else:
        subj_max_len, ques_max_len, comm_max_len, overlap_ft = max_len

    subject_idx = convert2indices(subject_tok, alphabet, dummy_word_idx, subj_max_len)
    question_idx = convert2indices(question_tok, alphabet, dummy_word_idx, ques_max_len)
    comment_idx = convert2indices(comment_tok, alphabet, dummy_word_idx, comm_max_len)

    unique_questions = get_unique_questions(relq_id, question_tok)
    unique_subjects = get_unique_questions(relq_id, subject_tok)
    documents = unique_questions + comment_tok + unique_subjects
    word2df = compute_dfs(documents)

    overlap_feat = overlap_features(question_tok, comment_tok, word2df)

    label = np.asarray([convert_label(x[-1].replace('\n', '').replace('\r', '')) for x in data_raw], dtype='int32')
    return relq_id, relc_id, subject_idx, question_idx, comment_idx, overlap_feat, label, max_len



