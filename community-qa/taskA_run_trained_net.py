import os
import cPickle
import json
from collections import defaultdict
from nltk import TweetTokenizer
from data_utils import load_data_taskA
from keras.models import model_from_json
from keras.utils.np_utils import probas_to_classes


model_path = 'model'
base_model_path = os.path.join(model_path, 'base_model.json')
trained_model_path = os.path.join(model_path, 'trained_models')
current_trained_model_path = os.path.join(trained_model_path, 'trained_weights_{}.h5'.format(1))
fname_train_orig = 'data_folder/semeval2016-task3-taskB-subsystemA-input-train.tsv'
fname_train_rel = 'data_folder/semeval2016-task3-taskA-train.tsv'
fname_test_orig = 'data_folder/semeval2016-task3-taskB-subsystemA-input-test.tsv'
fname_test_rel = 'data_folder/semeval2016-task3-taskA-test.tsv'

config = json.load(open(base_model_path))
ilayer_names = []
for input_layer in config['config']['input_layers']:
    ilayer_names.append(input_layer[0])

ilen = defaultdict(lambda : 0)
for layer in config['config']['layers']:
    name = layer['config']['name']
    if name in ilayer_names:
        ilenght = layer['config']['batch_input_shape'][1]
        ilen[name] = ilenght


print('Load Vocabulary')
alphabet = cPickle.load(open('vocabularies/vocab_en300M_reduced.pickle', 'rb'))
tknzr = TweetTokenizer()

print('Load Data')

max_subject_len = ilen['subject_input']
max_question_len = ilen['question_input']
max_comment_len = ilen['comment_input']
max_len = (max_subject_len, max_question_len, max_comment_len)

relq_id, relc_id, subject_idx, question_idx, comment_idx, subj_overlap_idx, ques_overlap_idx, comm_overlap_idx, overlap_feat, label = load_data_taskA(fname_train_rel, alphabet, tknzr, max_len=max_len)
relq_id_test, relc_id_test, subject_idx_test, question_idx_test, comment_idx_test, subj_overlap_idx_test, ques_overlap_idx_test, comm_overlap_idx_test, overlap_feat_test, label_test = load_data_taskA(fname_test_rel, alphabet, tknzr, max_len=max_len)

input_train = [
    subject_idx,
    question_idx,
    comment_idx,
    subj_overlap_idx,
    ques_overlap_idx,
    comm_overlap_idx,
    overlap_feat
]

input_test = [
    subject_idx_test,
    question_idx_test,
    comment_idx_test,
    subj_overlap_idx_test,
    ques_overlap_idx_test,
    comm_overlap_idx_test,
    overlap_feat_test
]


print('Load Trained Model')
model = model_from_json(open(base_model_path).read())
model.load_weights(current_trained_model_path)

y_pred_train = model.predict(input_train)
y_pred_test = model.predict(input_test)

y_pred_test_cls = probas_to_classes(y_pred_test)

ofile = open('data_folder/semeval2016-task3-taskA-pred.tsv', 'wb')
for i, (qid, cid) in enumerate(zip(relq_id_test, relc_id_test)):
    p = y_pred_test_cls[i]
    if p == 0:
        label = 'false'
    elif p == 1:
        label = 'true'
    proba = y_pred_test[i][1]

    outline = '{}\t{}\t{}\t{}\t{}\n'.format(qid,cid,0,proba,label)
    ofile.write(outline)
ofile.close()

train_results = defaultdict(lambda : [])
for i, rqid in enumerate(relq_id):
    train_results[rqid].append(y_pred_train[i][1])

test_results = defaultdict(lambda : [])
for i, rqid in enumerate(relq_id_test):
    test_results[rqid].append(y_pred_test[i][1])

ofile = open('data_folder/semeval2016-task3-taskB-subtaskA-output-train-rel.tsv', 'wb')
for rqid in sorted(train_results.keys()):
    oqid = rqid.split('_')[0]
    predictions = '\t'.join(map(lambda x: str(x), train_results[rqid]))

    outline = '{}\t{}\t{}\n'.format(oqid, rqid, predictions)
    ofile.write(outline)
ofile.close()

ofile = open('data_folder/semeval2016-task3-taskB-subtaskA-output-test-rel.tsv', 'wb')
for rqid in sorted(test_results.keys()):
    oqid = rqid.split('_')[0]
    predictions = '\t'.join(map(lambda x: str(x), test_results[rqid]))

    outline = '{}\t{}\t{}\n'.format(oqid, rqid, predictions)
    ofile.write(outline)
ofile.close()