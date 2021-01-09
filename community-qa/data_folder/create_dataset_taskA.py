from xml.etree import ElementTree
import sys
import codecs

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)

train_2016_part1 = 'SemEval2016-Task3-CQA-QL-train-part1.xml'
train_2016_part2 = 'SemEval2016-Task3-CQA-QL-train-part2.xml'
dev_2016 = 'SemEval2016-Task3-CQA-QL-dev.xml'
test_2016 = 'SemEval2016-Task3-CQA-QL-test.xml'

outfile = open('semeval2016-task3-taskB-subsystemA-input-test.tsv', 'wb')
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
        #print '{}\t{}\t{}'.format(related_question_id, rel_subject.encode('ascii', 'xmlcharrefreplace'),related_question.encode('ascii', 'xmlcharrefreplace'))

        #get all comments for realted question
        for rel_c in thread.findall('RelComment'):
            related_comment_id = rel_c.get('RELC_ID')
            related_comment = unicode(rel_c.find('RelCText').text)
            related_comment_label = rel_c.get('RELC_RELEVANCE2RELQ')
            #print '{}\t{}\t{}'.format(related_comment_id, related_comment.encode('ascii', 'xmlcharrefreplace'), related_comment_label)

            outline = u'{}\t{}\t{}\t{}\t{}\t{}'.format(related_question_id,related_comment_id,orig_question_subject,orig_question_text,related_comment,related_comment_label)
            print outline.encode('ascii', 'ignore')
            outfile.write(outline.encode('utf8') + '\n')

