__author__ = 'burness'
from collections import defaultdict, Counter
from glob import glob
import sys

glob_files = sys.argv[1]
loc_outfile = sys.argv[2]

def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
    if method == "average":
        scores = defaultdict(list)
    with open(loc_outfile,"wb") as outfile:
        # for i, glob_file in enumerate(filter_subs(glob(glob_files))):
        for i , glob_file in enumerate(glob(glob_files)):
            print "parsing:", glob_file
            print "i:" , i
            # sort glob_file by first column, ignoring the first line
            lines = open(glob_file).readlines()
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate( lines ):
                if i == 0 and e == 0:
                    outfile.write(line)
                if e > 0:
                    row = line.strip().split(",")
                    scores[(e,row[0])].append(row[1])
        for j,k in sorted(scores,key=lambda d:int(d[0])):
            outfile.write("%s,%s\n"%(k,Counter(scores[(j,k)]).most_common(1)[0][0]))
        print("wrote to %s"%loc_outfile)
def filter_subs(glo_files_list):
    result = []
    # with open('../data/result/result_stat.txt','r') as fread:
    with open('../data/logs_0119/7.txt','r') as fread:
        for line in fread.readlines():
            train_score_path = line.split(' ')
            train_score = train_score_path[0]
            path = train_score_path[1].strip()
            # print float(train_score)
            if float(train_score) > 0.792:
                result.append(path)
            #     continue
            # else:
            #     result.append(path)
        # print sub_stat.head()
    return result
print len(glob(glob_files))
print loc_outfile
# print glob(glob_files)
# filter the glob_files according to the training score
# results = filter_subs(glob(glob_files))
# print len(results)
# kaggle_bag(glob_files, loc_outfile)
kaggle_bag(glob_files,loc_outfile)


