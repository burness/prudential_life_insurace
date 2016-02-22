'''
Coding Just for Fun
Created by burness on 16/1/25.
'''
from collections import defaultdict, Counter
from glob import glob
import sys
from cv_topK import sort_get_topK
def kaggle_bag(loc_outfile, method="average", weights="uniform",K=100):
    if method == "average":
        scores = defaultdict(list)
    with open(loc_outfile,"wb") as outfile:
        # for i, glob_file in enumerate(filter_subs(glob(glob_files))):
        file_list = [ x for (x,y) in sort_get_topK('../data/logs_0124/final.txt',K=K)]
        print file_list
        for i , glob_file in enumerate(file_list):
            # print "parsing:", glob_file
            # print "i:" , i
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

if __name__ == "__main__":
    kaggle_bag('../data/result_0125_3.txt',K=3)