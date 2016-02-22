'''
Coding Just for Fun
Created by burness on 16/1/25.
data source: ../data/logs_0124/final.txt
sort the dict and get the top K
'''

def sort_get_topK(file_path, K=10):
    import re
    file_score = {}
    with open(file_path) as fread:
        for line in fread.readlines():
            match = re.search(r'file: ([0-7]).txt,params: (\d+) Train score is: (\S+)',line)
            folder = match.group(1)
            file = match.group(2)
            train_cv_score = match.group(3)
            submission_string = '../data/result_0119/'+folder+'/submission'+file+'.csv'
            # print submission_string
            file_score[submission_string] = train_cv_score
    # return file_score
    # print file_score
    return sorted(file_score.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:K]




# a=sort_get_topK('../data/logs_0124/final.txt',K=100)
# print a
