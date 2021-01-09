data_raw_relev = open('SemEval2016-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy', 'rb').readlines()

data_raw_pred = open('semeval2016-task3-taskA-test.tsv', 'rb').readlines()

data_id_relevant = map(lambda x: x.replace('\n', '').split('\t'), data_raw_relev)
data_id_pred = map(lambda x: x.replace('\n', '').split('\t'), data_raw_pred)

rel_id = set(map(lambda x: x[0], data_id_relevant))

data_id_pred_fil = filter(lambda x: x[0] in rel_id, data_id_pred)
#data_id_pred_fil = map(lambda x: [x[0], x[1], x[2], str(float(x[3])), x[4]], data_id_pred_fil)

data_pred = map(lambda x: '\t'.join(x) + '\n', data_id_pred_fil)

open('semeval2016-task3-taskA-test-filter.tsv', 'wb').writelines(data_pred)