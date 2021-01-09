from keras.layers import Input, merge, Convolution2D, ZeroPadding1D, MaxPooling2D, Flatten, Dense, Layer
from architectures.attention_layers import Level1AttentionTransformation, level1_attention
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
import keras.backend as K


def expand_dims(x):
    return K.expand_dims(x, dim=-1)


def expand_dims_output_shape(input_shape):
    sh_list = list(input_shape)
    sh_list.append(1)
    return tuple(sh_list)


def build_model_abcnn1(embeddings_words, embedding_dim, max_len, tag):
    max_question_len = max_len[1]
    max_comment_len = max_len[2]
    overlap_ft_len = max_len[3]

    nb_filter = 200
    filter_length = 5

    input_comment_idx = Input(batch_shape=(None, max_comment_len), dtype='int32', name='comment_input_{}'.format(tag))
    #input_subj_idx = Input(batch_shape=(None, max_subject_len), dtype='int32', name='subject_input')
    input_quest_idx = Input(batch_shape=(None, max_question_len), dtype='int32', name='question_input')

    #emb_subj_idx = embeddings_words(input_subj_idx)
    emb_ques_idx = embeddings_words(input_quest_idx)
    emb_comm_idx = embeddings_words(input_comment_idx)

    normalized_ques_emb = BatchNormalization(mode=1)(emb_ques_idx)
    normalized_comm_emb = BatchNormalization(mode=1)(emb_comm_idx)

    attention_layer = level1_attention(layers=[emb_ques_idx, emb_comm_idx], name='Attention Layer')
    attention_question_trafo = Level1AttentionTransformation(transpose=False, output_dim=embedding_dim)(attention_layer)
    attention_comment_trafo = Level1AttentionTransformation(transpose=True, output_dim=embedding_dim)(attention_layer)

    zeropadding_quest = ZeroPadding1D(filter_length - 1)(normalized_ques_emb)
    zeropadding_comm = ZeroPadding1D(filter_length - 1)(normalized_comm_emb)
    zeropadding_quest_trafo = ZeroPadding1D(filter_length - 1)(attention_question_trafo)
    zeropadding_comm_trafo = ZeroPadding1D(filter_length - 1)(attention_comment_trafo)

    question_emb_ext = Lambda(expand_dims, expand_dims_output_shape)(zeropadding_quest)
    comment_emb_ext = Lambda(expand_dims, expand_dims_output_shape)(zeropadding_comm)
    qtrafo_emb_ext = Lambda(expand_dims, expand_dims_output_shape)(zeropadding_quest_trafo)
    ctrafo_emb_ext = Lambda(expand_dims, expand_dims_output_shape)(zeropadding_comm_trafo)

    question_merge = merge(inputs=[question_emb_ext, qtrafo_emb_ext], mode='concat', dot_axes=-1)
    comment_merge = merge(inputs=[comment_emb_ext, ctrafo_emb_ext], mode='concat', dot_axes=-1)

    conv_quest = Convolution2D(
        nb_filter=nb_filter,
        nb_row=filter_length,
        nb_col=embedding_dim,
        input_shape=(zeropadding_quest._keras_shape[0], embedding_dim, 2),
        border_mode='valid',
        activation='tanh',
        dim_ordering='tf'
        # W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(question_merge)

    conv_comm = Convolution2D(
        nb_filter=nb_filter,
        nb_row=filter_length,
        nb_col=embedding_dim,
        input_shape=(zeropadding_quest._keras_shape[0], embedding_dim, 2),
        border_mode='valid',
        activation='tanh',
        dim_ordering='tf'
        # W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(comment_merge)

    maxpool_quest = MaxPooling2D(
        pool_size=(conv_quest._keras_shape[1], 1),
        dim_ordering='tf'
    )(conv_quest)
    maxpool_comm = MaxPooling2D(
        pool_size=(conv_comm._keras_shape[1], 1),
        dim_ordering='tf'
    )(conv_comm)

    flat_quest = Flatten()(maxpool_quest)
    flat_comm = Flatten()(maxpool_comm)

    sim_quest_comm = merge([flat_quest, flat_comm], mode='cos', dot_axes=-1,
                           name='merge_question_comment_{}'.format(tag))

    flat_sim_quest_comm = Flatten()(sim_quest_comm)

    merge_all = merge([flat_quest, flat_comm, flat_sim_quest_comm],
                      mode='concat', concat_axis=-1)

    hidden = Dense(
        2 * nb_filter + 1,
        activation='relu'
        # W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(merge_all)

    softmax = Dense(2, activation='softmax')(hidden)

    ret_inputs = [
        input_quest_idx,
        input_comment_idx
    ]

    return ret_inputs, softmax, flat_sim_quest_comm, softmax


