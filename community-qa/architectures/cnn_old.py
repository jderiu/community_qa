from keras.layers import Input, merge, Convolution1D, ZeroPadding1D, MaxPooling1D, Flatten, Dense, Layer

def build_model_cnn(emb_subj_idx, emb_ques_idx, embeddings_words, embeddings_overlap, max_len, tag):
    max_subject_len = max_len[0]
    max_question_len = max_len[1]
    max_comment_len = max_len[2]
    overlap_ft_len = max_len[3]

    input_comment_idx = Input(batch_shape=(None, max_comment_len), dtype='int32', name='comment_input_{}'.format(tag))

    input_qoverlap = Input(batch_shape=(None, max_question_len), dtype='int32', name='qoverlap_input_{}'.format(tag))
    input_coverlap = Input(batch_shape=(None, max_comment_len), dtype='int32', name='coverlap_input_{}'.format(tag))
    input_soverlap = Input(batch_shape=(None, max_subject_len), dtype='int32', name='soverlap_input_{}'.format(tag))

    input_overlap_ft = Input(batch_shape=(None, overlap_ft_len), dtype='float32', name='overlap_features_{}'.format(tag))
    emb_comm_idx = embeddings_words(input_comment_idx)

    emb_qoverlap = embeddings_overlap(input_qoverlap)
    emb_coverlap = embeddings_overlap(input_coverlap)
    emb_soverlap = embeddings_overlap(input_soverlap)

    merge_subject = merge([emb_subj_idx, emb_soverlap], mode='concat', concat_axis=-1)
    merge_questions = merge([emb_ques_idx, emb_qoverlap], mode='concat', concat_axis=-1)
    merge_comments = merge([emb_comm_idx, emb_coverlap], mode='concat', concat_axis=-1)

    nb_filter = 200
    filter_length = 5

    zeropadding_subj = ZeroPadding1D(filter_length - 1)(merge_subject)
    zeropadding_quest = ZeroPadding1D(filter_length - 1)(merge_questions)
    zeropadding_comm = ZeroPadding1D(filter_length - 1)(merge_comments)

    conv_subj = Convolution1D(
        nb_filter=nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1,
        #W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(zeropadding_subj)

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

    maxpool_subj = MaxPooling1D(pool_length=conv_subj._keras_shape[1])(conv_subj)
    maxpool_quest = MaxPooling1D(pool_length=conv_quest._keras_shape[1])(conv_quest)
    maxpool_comm = MaxPooling1D(pool_length=conv_comm._keras_shape[1])(conv_comm)

    flat_subj = Flatten()(maxpool_subj)
    flat_quest = Flatten()(maxpool_quest)
    flat_comm = Flatten()(maxpool_comm)

    sim_sub_comm = merge([flat_subj, flat_comm], mode='cos', dot_axes=-1, name='merge_subject_comment_{}'.format(tag))
    sim_quest_comm = merge([flat_quest, flat_comm], mode='cos', dot_axes=-1, name='merge_question_comment_{}'.format(tag))

    flat_sim_sub_comm = Flatten()(sim_sub_comm)
    flat_sim_quest_comm = Flatten()(sim_quest_comm)

    merge_all = merge([flat_subj, flat_quest, flat_comm, input_overlap_ft, flat_sim_sub_comm, flat_sim_quest_comm],
                      mode='concat', concat_axis=-1)

    hidden = Dense(
        3 * nb_filter + 2 + overlap_ft_len,
        #W_regularizer=l1l2(l1=0.01, l2=0.01)
    )(merge_all)
    softmax = Dense(2, activation='softmax')(hidden)

    ret_inputs = [
        input_comment_idx,
        input_soverlap,
        input_qoverlap,
        input_coverlap,
        input_overlap_ft
    ]

    return ret_inputs, softmax, flat_sim_quest_comm, flat_sim_sub_comm