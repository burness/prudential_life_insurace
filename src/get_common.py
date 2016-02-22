'''
Coding Just for Fun
Created by burness on 16/2/10.
'''

def get_common(file1, file2):
    dict1 = {}
    with open(file1,'r') as f1:
        for line_no, line in enumerate(f1.readlines()):
            if line_no == 0:
                continue
            else:
                key = line.split(',')[0]
                val = line.split(',')[1]
                dict1[key] = val
    dict2 = {}
    with open(file2,'r') as f2:
        for line_no, line in enumerate(f2.readlines()):
            if line_no == 0:
                continue
            else:
                key = line.split(',')[0]
                val = line.split(',')[1]
                dict2[key] = val

    dict_common = {}
    dict_uncommon = {}
    for k, v in dict1.items():
        if dict1[k] == dict2[k]:
            dict_common[k] = v
        else:

            dict_uncommon[k] = [v, dict2[k]]
    return dict_common, dict_uncommon

if __name__ == '__main__':
    # dict_common, dict_uncommon = get_common('voting_0210_5.csv','voting_0203_1.csv')
    dict_common, dict_uncommon = get_common('voting_0210_1.csv','voting_0131_3.csv')
    print len(dict_common), len(dict_uncommon)
    print dict_uncommon
    # judge the uncommon part according to voting by all the submission above 0.67400
    dict3 = {}
    with open('tmp_voting_0215_1.csv','r') as f3:
        for line_no, line in enumerate(f3.readlines()):
            if line_no == 0:
                continue
            else:
                key = line.split(',')[0]
                val = line.split(',')[1]
                dict3[key] = val


    for k,v in dict_uncommon.items():
        dict_common[k] = dict3[k]
    print len(dict_common)

    with open('voting_0215_2.csv','w') as fw:
        fw.write('Id,Response\n')
        # for k,v in dict_common.items():
        for k, v in sorted(dict_common.items(),key=lambda d:int(d[0])):
            line = k+','+v
            fw.write(line)


