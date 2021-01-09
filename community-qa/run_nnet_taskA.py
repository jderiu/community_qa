import sys
import json
import os

import keras.backend as K
import numpy as np
from data_utils import load_data_taskA
from evaluation_metrics import mean_average_precision, f1_score_task3
from gensim.models import Phrases
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical, probas_to_classes
from keras.optimizers import Adadelta, Adagrad, SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.regularizers import l1l2
from architectures.abcnn import build_model_abcnn1
from architectures.abcnn3 import build_model_abcnn3
from custom_callbacks.LayerTracker import LayerTracker
from keras.objectives import categorical_crossentropy
from nltk.tokenize import TweetTokenizer
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle


def load_transformer_list():
    output_directory = 'vocabularies'
    output_basename = 'en_embeddings_200M_200d'

    path = os.path.join(output_directory, output_basename)
    config_fname = os.path.join(path, 'config.json')
    with open(config_fname, 'r') as json_data:
        wemb_config = json.load(json_data)
        ngrams = wemb_config['ngrams']

        transformers = []
        for i in range(ngrams - 1):
            phrase_model = Phrases.load(os.path.join(path, '{}gram'.format(i)))
            transformers.append(phrase_model)

    return transformers

np.set_printoptions(threshold=np.nan)
model_path = 'model/taskA'
base_model_path = os.path.join(model_path, 'base_model.json')
base_weights_path = os.path.join(model_path, 'base_weights.h5')
trained_model_path = os.path.join(model_path, 'trained_models')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(trained_model_path):
    os.makedirs(trained_model_path)

print('Load Embeddings')
fname_wordembeddings = os.path.join('vocabularies/en_embeddings_200M_200d', 'embedding_matrix.npy')
vocab_emb = np.load(fname_wordembeddings)
print('Word embedding matrix size: {}'.format(vocab_emb.shape))
max_features = vocab_emb.shape[0]
embedding_dims = vocab_emb.shape[1]

print('Load Vocabulary')
fname = 'data_folder/semeval2016-task3-taskA-train-cleaned.tsv'
fname_test = 'data_folder/semeval2016-task3-taskA-test-filter.tsv'
alphabet = cPickle.load(open('vocabularies/en_embeddings_200M_200d/vocabulary.pickle', 'rb'))
print('Vocabulary Size: {}'.format(len(alphabet)))
tknzr = TweetTokenizer()

print('Load Transofmers')
transformers = load_transformer_list()

print('Load Data')
relq_id, relc_id, subject_idx, question_idx, comment_idx, overlap_feat, label, max_len_pos = load_data_taskA(fname, alphabet, tknzr, transformers)

max_subject_len = max_len_pos[0]
max_question_len = max_len_pos[1]
max_comment_len = max_len_pos[2]
overlap_ft_len = max_len_pos[3]

#subj, question, comment+, comment-, soverlap+, qoverap+, commoverlap1+, soverlap-, qoverap-, commoverlap1-, oft+, oft-
#loaded_data = load_data_taskA_pair(fname, alphabet, tknzr)

relq_id_test, relc_id_test, subject_idx_test, question_idx_test, comment_idx_test, overlap_feat_test, label_test, max_len_pos = load_data_taskA(fname_test, alphabet, tknzr, transformers, max_len=max_len_pos)
print('Number of Test Samples: {}'.format(label_test.shape[0]))
print(np.bincount(label_test))

print('Subj Len: {}\tQuestion Len: {}\tComment Len: {}'.format(max_subject_len, max_question_len, max_comment_len))

print('Build Model')
#max_len_pos = (31, 121, 299, 2)
embeddings_words = Embedding(output_dim=embedding_dims, input_dim=max_features, weights=[vocab_emb], name='zshared_embeddings', trainable=False)
#embeddings_words = Embedding(output_dim=embedding_dims, input_dim=max_features, name='zshared_embeddings')

ret_inputs_pos, softmax_pos, flat_sim_quest_comm, attention_layer = build_model_abcnn3(embeddings_words, embedding_dims, max_len_pos, 'abcnn')

keras_inputs_train = ret_inputs_pos

train_model = Model(input=keras_inputs_train, output=softmax_pos)
test_model = Model(input=keras_inputs_train, output=attention_layer)
train_model.summary()
optimizer = Adadelta(lr=0.005, rho=0.95, epsilon=1e-8)
#optimizer = Adagrad(lr=0.0001, epsilon=1e-8, decay=0.0)
#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=0.0001)
#plot(model, to_file='model.png')
train_model.compile(loss=['categorical_crossentropy'], optimizer=optimizer, metrics=[f1_score_task3])

test_model.compile(loss=['categorical_crossentropy'], optimizer=optimizer, metrics=[f1_score_task3])

input_train = [
    question_idx,
    comment_idx,
]


output_train = [
    to_categorical(label, 2),
    #label
]

input_test = [
    question_idx_test,
    comment_idx_test,
]

output_test = [
    to_categorical(label_test, 2),
    #label_test
]

current_trained_model_path = os.path.join(trained_model_path, 'trained_weights_{}.h5'.format(1))

early_stop = EarlyStopping(
    monitor='val_f1_score_task3',
    patience=20,
    verbose=1,
    mode='max')

model_checkpoit = ModelCheckpoint(
    filepath=current_trained_model_path,
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_f1_score_task3',
    mode='max')

#nan_detector = NaNDetector()
layer_tracker = LayerTracker(0)

hist = train_model.fit(
    input_train,
    output_train,
    validation_data=(input_test, output_test),
    batch_size=100,
    nb_epoch=1000,
    callbacks=[model_checkpoit, early_stop],
    shuffle=True
)

print( 'Load Best Net')
train_model.load_weights(current_trained_model_path)
print('Loaded Best Net')
y_pred = train_model.predict(input_test)
y_pred_test_cls = probas_to_classes(y_pred)
print('Predicted')

y_true = K.variable(value=label_test)
y_pred_keras = K.variable(value=y_pred_test_cls)
print(K.eval(mean_average_precision(y_true, y_pred_keras)))

loss_over_time = hist.history['loss']
val_loss_over_time = hist.history['val_f1_score_task3']
np.save(open('results/loss.npy', 'w'), loss_over_time)
np.save(open('results/val_loss.npy', 'w'), val_loss_over_time)

ofile1 = open('data_folder/semeval2016-task3-taskA-pred.tsv', 'wb')
for i, (qid, cid) in enumerate(zip(relq_id_test, relc_id_test)):
    p = y_pred_test_cls[i]
    if p == 0:
        label = 'false'
    elif p == 1:
        label = 'true'
    proba1 = y_pred[i][1]
    outline1 = '{}\t{}\t{}\t{}\t{}\n'.format(qid,cid,0,proba1,label)
    ofile1.write(outline1)
ofile1.close()