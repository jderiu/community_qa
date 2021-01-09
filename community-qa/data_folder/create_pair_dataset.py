
ifile = open('semeval2016-task3-taskA-train-cleaned.tsv', 'rb')
ofile = open('semeval2016-task3-taskA-train-pair.tsv', 'wb')

idata = ifile.readlines()
idata_split = map(lambda x: x.replace('\n','').split('\t'), idata)
