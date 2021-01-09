from data_utils import load_data_taskB
import cPickle
from nltk.tokenize import TweetTokenizer
import os
import numpy as np
from evaluation_metrics import mean_average_precision, f1_score_keras, f1_score_task3, precision_score_task3
from sklearn.cross_validation import KFold
from keras.layers import Input, Embedding, merge, Convolution1D, ZeroPadding1D, MaxPooling1D, Flatten, Dense
from keras.models import Model
from keras.utils.np_utils import to_categorical, probas_to_classes
from keras.models import model_from_json
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.regularizers import l1l2
#from keras.utils.visualize_util import plot

model_path = 'model/taskB'
base_model_path = os.path.join(model_path, 'base_model.json')
base_weights_path = os.path.join(model_path, 'base_weights.h5')
trained_model_path = os.path.join(model_path, 'trained_models')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(trained_model_path):
    os.makedirs(trained_model_path)

print('Load Embeddings')
fname_wordembeddings = os.path.join('vocabularies', 'emb_smiley_tweets_embedding_english_590M.npy')
vocab_emb = np.load(fname_wordembeddings)
print('Word embedding matrix size: {}'.format(vocab_emb.shape))
max_features = vocab_emb.shape[0]
embedding_dims = vocab_emb.shape[1]

print('Load Vocabulary')
fname = 'data_folder/semeval2016-task3-taskB-train.tsv'
fname_test = 'data_folder/semeval2016-task3-taskB-test.tsv'
alphabet = cPickle.load(open('vocabularies/vocab_en300M_reduced.pickle', 'rb'))
print('Vocabulary Size: {}'.format(len(alphabet)))
tknzr = TweetTokenizer()

print('Load Data')
orig_id, relq_id, orig_subject_idx, rel_subject_idx, orig_text_idx, rel_text_idx, sorig_overlap_ss, srel_overlap_ss, orig_overlap_st_tt, rel_overlap_st_tt, qorig_meta, qrel_meta, overlap_feat, label = load_data_taskB(fname, alphabet, tknzr)

max_osubject_len = orig_subject_idx.shape[1]
max_rsubject_len = rel_subject_idx.shape[1]
max_oquestion_len = orig_text_idx.shape[1]
max_rquestion_len = rel_text_idx.shape[1]

max_len = (max_osubject_len, max_rsubject_len, max_oquestion_len, max_rquestion_len)

orig_id_test, relq_id_test, orig_subject_idx_test, rel_subject_idx_test, orig_text_idx_test, rel_text_idx_test, sorig_overlap_ss_test, srel_overlap_ss_test, orig_overlap_st_tt_test, rel_overlap_st_tt_test, qorig_meta_test, qrel_meta_test, overlap_feat_test, label_test = load_data_taskB(fname_test, alphabet, tknzr, max_len=max_len)

print('Number of Train Samples: {}'.format(label.shape[0]))
print('Number of Test Samples: {}'.format(label_test.shape[0]))

overlap_ft_len = overlap_feat.shape[1]
meta_len = qrel_meta.shape[1]

assert orig_subject_idx.shape[0] == rel_subject_idx.shape[0] == orig_text_idx.shape[0] == rel_text_idx.shape[0]
assert orig_subject_idx.shape == sorig_overlap_ss.shape
assert rel_subject_idx.shape == srel_overlap_ss.shape
assert orig_text_idx.shape == orig_overlap_st_tt.shape
assert rel_text_idx.shape == rel_overlap_st_tt.shape
assert qrel_meta.shape == qorig_meta.shape


print('Orig Subj Len: {}\tRel Subj Len: {}\tOrig Question Len: {}\tRel Question Len: {}'.format(max_osubject_len, max_rsubject_len, max_oquestion_len, max_rquestion_len))
print('Build Model')

regularizer_c1 = l1l2(l1=0.01, l2=0.01)
regularizer_c2 = l1l2(l1=0.01, l2=0.01)
regularizer_c3 = l1l2(l1=0.01, l2=0.01)
regularizer_c4 = l1l2(l1=0.01, l2=0.01)
regularizer = l1l2(l1=0.01, l2=0.01)

input_osubj_idx = Input(batch_shape=(None, max_osubject_len), dtype='int32', name='orig_subject_input')
input_rsubj_idx = Input(batch_shape=(None, max_rsubject_len), dtype='int32', name='rel_subject_input')
input_otext_idx = Input(batch_shape=(None, max_oquestion_len), dtype='int32', name='orig_text_input')
input_rtext_idx = Input(batch_shape=(None, max_rquestion_len), dtype='int32', name='rel_text_input')

input_osubj_overlap = Input(batch_shape=(None, max_osubject_len), dtype='int32', name='orig_subject_overlap')
input_rsubj_overlap = Input(batch_shape=(None, max_rsubject_len), dtype='int32', name='rel_subject_overlap')
input_otext_overlap = Input(batch_shape=(None, max_oquestion_len), dtype='int32', name='orig_text_overlap')
input_rtext_overlap = Input(batch_shape=(None, max_rquestion_len), dtype='int32', name='rel_text_overlap')

input_overlap_ft = Input(batch_shape=(None, overlap_ft_len), dtype='float32', name='overlap_features')
#input_ometa = Input(batch_shape=(None, meta_len), dtype='float32', name='orig_meta_info')
#input_rmeta = Input(batch_shape=(None, meta_len), dtype='float32', name='rel_meta_info')

embeddings_words = Embedding(
        output_dim= embedding_dims,
        input_dim=max_features,
        weights=[vocab_emb],
        name='zshared_embeddings',
)

emb_osubj_idx = embeddings_words(input_osubj_idx)
emb_rsubj_idx = embeddings_words(input_rsubj_idx)
emb_otext_idx = embeddings_words(input_otext_idx)
emb_rtext_idx = embeddings_words(input_rtext_idx)

overlap_dims = 5
overlap_ft_st_tt = np.max(orig_overlap_st_tt) + 1
overlap_ft_ss = np.max(sorig_overlap_ss) + 1
embeddings_overlap_ss = Embedding(overlap_ft_ss, overlap_dims, init='lecun_uniform')
embeddings_overlap_st_tt = Embedding(overlap_ft_st_tt, overlap_dims, init='lecun_uniform')

emb_osubj_overlap = embeddings_overlap_ss(input_osubj_overlap)
emb_rsubj_overlap = embeddings_overlap_ss(input_rsubj_overlap)
emb_otext_overlap = embeddings_overlap_st_tt(input_otext_overlap)
emb_rtext_overlap = embeddings_overlap_st_tt(input_rtext_overlap)

merge_osubj = merge([emb_osubj_idx, emb_osubj_overlap], mode='concat', concat_axis=-1)
merge_rsubj = merge([emb_rsubj_idx, emb_rsubj_overlap], mode='concat', concat_axis=-1)
merge_otext = merge([emb_otext_idx, emb_otext_overlap], mode='concat', concat_axis=-1)
merge_rtext = merge([emb_rtext_idx, emb_rtext_overlap], mode='concat', concat_axis=-1)

nb_filter = 200
filter_length = 5

zeropadding_osubj = ZeroPadding1D(filter_length-1)(merge_osubj)
zeropadding_rsubj = ZeroPadding1D(filter_length-1)(merge_rsubj)
zeropadding_otext = ZeroPadding1D(filter_length-1)(merge_otext)
zeropadding_rtext = ZeroPadding1D(filter_length-1)(merge_rtext)

conv_osubj = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1,
    #W_regularizer=regularizer_c1
)(zeropadding_osubj)

conv_rsubj = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1,
    #W_regularizer=regularizer_c2
)(zeropadding_rsubj)

conv_otext = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1,
    #W_regularizer=regularizer_c3
)(zeropadding_otext)

conv_rtext = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1,
    #W_regularizer=regularizer_c4
)(zeropadding_rtext)

maxpool_osubj = MaxPooling1D(pool_length=conv_osubj._keras_shape[1])(conv_osubj)
maxpool_rsubj = MaxPooling1D(pool_length=conv_rsubj._keras_shape[1])(conv_rsubj)
maxpool_otext = MaxPooling1D(pool_length=conv_otext._keras_shape[1])(conv_otext)
maxpool_rtext = MaxPooling1D(pool_length=conv_rtext._keras_shape[1])(conv_rtext)

flat_osubj = Flatten()(maxpool_osubj)
flat_rsubj = Flatten()(maxpool_rsubj)
flat_otext = Flatten()(maxpool_otext)
flat_rtext = Flatten()(maxpool_rtext)

sim_subj = merge([flat_osubj, flat_rsubj], mode='cos', dot_axes=1)
sim_text = merge([flat_otext, flat_rtext], mode='cos', dot_axes=1)

flat_subj_sim = Flatten()(sim_subj)
flat_text_sim = Flatten()(sim_text)

#meta_sim = merge([input_ometa, input_rmeta], mode='cos', dot_axes=1)
#flat_meta_sim = Flatten()(meta_sim)

merge_all = merge([flat_osubj, flat_rsubj, flat_otext, flat_rtext, input_overlap_ft, flat_subj_sim, flat_text_sim], mode='concat', concat_axis=-1)

hidden = Dense(3*nb_filter + 2 + overlap_dims)(merge_all)
softmax = Dense(2, activation='softmax')(hidden)


inputs = [
    input_osubj_idx,
    input_rsubj_idx,
    input_otext_idx,
    input_rtext_idx,
    input_osubj_overlap,
    input_rsubj_overlap,
    input_otext_overlap,
    input_rtext_overlap,
    input_overlap_ft,
    #input_ometa,
    #input_rmeta
]

model = Model(input=inputs, output=[softmax])
#model.summary()
adadelta = Adadelta(lr=0.9, rho=0.95, epsilon=1e-9)
#plot(model, to_file='model_taskB.png')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Store Untrained Model')
json_string = model.to_json()
open(base_model_path, 'w').write(json_string)
model.save_weights(base_weights_path, overwrite=True)

input_train = [
    orig_subject_idx,
    rel_subject_idx,
    orig_text_idx,
    rel_text_idx,
    sorig_overlap_ss,
    srel_overlap_ss,
    orig_overlap_st_tt,
    rel_overlap_st_tt,
    overlap_feat,
    #qorig_meta,
    #qrel_meta
]

input_test = [
    orig_subject_idx_test,
    rel_subject_idx_test,
    orig_text_idx_test,
    rel_text_idx_test,
    sorig_overlap_ss_test,
    srel_overlap_ss_test,
    orig_overlap_st_tt_test,
    rel_overlap_st_tt_test,
    overlap_feat_test,
    #qorig_meta_test,
    #qrel_meta_test
]

print('Load Distant  Model')
model = model_from_json(open(base_model_path).read())
model.load_weights(base_weights_path)
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=[f1_score_task3])

current_trained_model_path = os.path.join(trained_model_path, 'trained_weights_{}.h5'.format(1))

early_stop = EarlyStopping(
    monitor='val_f1_score_task3',
    patience=100,
    verbose=1,
    mode='max')

model_checkpoit = ModelCheckpoint(
    filepath=current_trained_model_path,
    verbose=1,
    save_best_only=True,
    monitor='val_f1_score_task3',
    mode='max')

hist = model.fit(
    input_train,
    to_categorical(label, 2),
    batch_size=1000,
    nb_epoch=1000,
    validation_data=(input_test, to_categorical(label_test, 2)),
    callbacks=[early_stop, model_checkpoit],
    #class_weight={0: 1.0, 1: 1.3}
)

model.load_weights(current_trained_model_path)
y_pred = model.predict(input_test)

y_true = K.variable(value=label_test)
y_pred_keras = K.variable(value=y_pred[:,1])
print K.eval(mean_average_precision(y_true, y_pred_keras))

loss_over_time = hist.history['loss']
val_loss_over_time = hist.history['val_loss']
np.save(open('results/loss.npy', 'w'), loss_over_time)
np.save(open('results/val_loss.npy', 'w'), val_loss_over_time)

y_pred_test_cls = probas_to_classes(y_pred)

ofile = open('data_folder/semeval2016-task3-taskB-pred.tsv', 'wb')
for i, (qid, cid) in enumerate(zip(orig_id_test, relq_id_test)):
    p = y_pred_test_cls[i]
    if p == 0:
        label = 'false'
    elif p == 1:
        label = 'true'
    proba = y_pred[i][1]

    outline = '{}\t{}\t{}\t{}\t{}\n'.format(qid,cid,0,proba,label)
    ofile.write(outline)
ofile.close()
