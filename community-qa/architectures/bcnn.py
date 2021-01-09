from keras.layers import Input, merge, Convolution1D, ZeroPadding1D, MaxPooling1D, Flatten, Dense, Layer

def build_model_cnn(embeddings_words, embedding_dim, max_len, tag):
    max_subject_len = max_len[0]
    max_question_len = max_len[1]
    max_comment_len = max_len[2]
    overlap_ft_len = max_len[3]

    input_quest_idx = Input(batch_shape=(None, max_question_len), dtype='int32', name='question_input')
    input_comment_idx = Input(batch_shape=(None, max_comment_len), dtype='int32', name='comment_input_{}'.format(tag))
    emb_comm_idx = embeddings_words(input_comment_idx)
    emb_ques_idx = embeddings_words(input_quest_idx)

    nb_filter = 200
    filter_length = 5

    zeropadding_quest = ZeroPadding1D(filter_length - 1)(emb_ques_idx)
    zeropadding_comm = ZeroPadding1D(filter_length - 1)(emb_comm_idx)

    conv_quest = Convolution1D(
        nb_filter=nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1,
        #W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(zeropadding_quest)

    conv_comm = Convolution1D(
        nb_filter=nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1,
        #W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(zeropadding_comm)

    maxpool_quest = MaxPooling1D(pool_length=conv_quest._keras_shape[1])(conv_quest)
    maxpool_comm = MaxPooling1D(pool_length=conv_comm._keras_shape[1])(conv_comm)

    flat_quest = Flatten()(maxpool_quest)
    flat_comm = Flatten()(maxpool_comm)

    sim_quest_comm = merge([flat_quest, flat_comm], mode='cos', dot_axes=-1, name='merge_question_comment_{}'.format(tag))

    flat_sim_quest_comm = Flatten()(sim_quest_comm)

    merge_all = merge([ flat_quest, flat_comm, flat_sim_quest_comm],
                      mode='concat', concat_axis=-1)

    hidden = Dense(
        2 * nb_filter + 1,
        #W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(merge_all)
    softmax = Dense(2, activation='softmax')(hidden)

    ret_inputs = [
        input_quest_idx,
        input_comment_idx
    ]

    return ret_inputs, softmax, flat_sim_quest_comm