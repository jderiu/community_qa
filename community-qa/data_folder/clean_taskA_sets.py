from collections import defaultdict
import random
import itertools

ifile = open('semeval2016-task3-taskA-train.tsv', 'rb').readlines()
ofile = open('semeval2016-task3-taskA-train-pairs.tsv', 'wb')
data_split = map(lambda x: x.replace('\r', '').replace('\n', '').split('\t'), ifile)
question_dict = defaultdict(lambda: [])

for line in data_split:
    qid = line[0]
    question_dict[qid].append(line)

counter = 0
for qid in question_dict.keys():
    good_comments = filter(lambda x: not x[-1] == 'Bad', question_dict[qid])
    bad_comments = filter(lambda x: x[-1] == 'Bad', question_dict[qid])

    for pos_line, neg_line in itertools.product(good_comments, bad_comments):
        oline = '\t'.join([pos_line[0], pos_line[1], pos_line[2], pos_line[3], pos_line[4], neg_line[4], pos_line[-1]]) + '\n'
        ofile.write(oline)
        oline = '\t'.join([pos_line[0], pos_line[1], pos_line[2], pos_line[3], neg_line[4], pos_line[4], neg_line[-1]]) + '\n'
        ofile.write(oline)
        counter += 1

print counter


