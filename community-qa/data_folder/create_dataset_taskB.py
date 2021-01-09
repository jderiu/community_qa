from xml.etree import ElementTree
import sys
import codecs
from collections import defaultdict

def create_meta_dict(fname):
    data = codecs.open(fname, 'rb', 'UTF-8').readlines()
    data_split = map(lambda x: x.replace('\n', '').split('\t'), data)
    data_dict = dict(map(lambda x: (x[1], '\t'.join(x[2:])), data_split))
    return data_dict

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)

train_2016_part1 = 'SemEval2016-Task3-CQA-QL-train-part1.xml'
train_2016_part2 = 'SemEval2016-Task3-CQA-QL-train-part2.xml'
dev_2016 = 'SemEval2016-Task3-CQA-QL-dev.xml'
test_2016 = 'SemEval2016-Task3-CQA-QL-test.xml'

train_2016_taskA_meta_rel = 'semeval2016-task3-taskB-subtaskA-output-train-rel.tsv'
train_2016_taskA_meta_orig = 'semeval2016-task3-taskB-subtaskA-output-train-orig.tsv'

test_2016_taskA_meta_rel = 'semeval2016-task3-taskB-subtaskA-output-test-rel.tsv'
test_2016_taskA_meta_orig = 'semeval2016-task3-taskB-subtaskA-output-test-orig.tsv'

train_2016_meta_dict_rel = create_meta_dict(train_2016_taskA_meta_rel)
train_2016_meta_dict_orig = create_meta_dict(train_2016_taskA_meta_orig)
test_2016_meta_dict_rel = create_meta_dict(test_2016_taskA_meta_rel)
test_2016_meta_dict_orig = create_meta_dict(test_2016_taskA_meta_orig)

outfile = open('semeval2016-task3-taskB-test.tsv', 'wb')
for fname in [test_2016]:
    e = ElementTree.parse(fname).getroot()
    for orig_question in e.findall('OrgQuestion'):
        orig_question_id = orig_question.get('ORGQ_ID')
        print(orig_question_id)
        thread = orig_question.find('Thread')
        orig_question_subject = unicode(orig_question.find('OrgQSubject').text)
        orig_question_text = unicode(orig_question.find('OrgQBody').text)

        #only 1 rel question
        rel_q = thread.find('RelQuestion')
        rel_subject = unicode(rel_q.find('RelQSubject').text)
        related_question = unicode(rel_q.find('RelQBody').text)
        related_question_id = rel_q.get('RELQ_ID')

        label = unicode(rel_q.get('RELQ_RELEVANCE2ORGQ'))

        orig_q_meta = test_2016_meta_dict_orig[related_question_id]
        rel_q_meta = test_2016_meta_dict_rel[related_question_id]

        outline = u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(orig_question_id, related_question_id, orig_question_subject,
                                                   rel_subject, orig_question_text, related_question, orig_q_meta, rel_q_meta, label)
        print outline.encode('ascii', 'ignore')
        outfile.write(outline.encode('utf8') + '\n')

