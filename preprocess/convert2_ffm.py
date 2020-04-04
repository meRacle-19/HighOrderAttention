
# coding: utf-8

# In[1]:


import sys, csv, math, gc, os
from collections import defaultdict
import pandas as pd


# In[2]:


def sample(input_path, train_path, eval_path, test_path, gen_file=True, debug=None, cols=[]):
    source = pd.read_csv(input_path, nrows=debug, header=None, names=cols, sep='\t')
    source = source.sample(frac=1.0)  # 全部打乱
    cut_idx1 = int(round(0.09 * source.shape[0]))
    cut_idx2 = int(round(0.1 * source.shape[0]))

    df_test, df_eval, df_train = source.iloc[:cut_idx1], source.iloc[cut_idx1:cut_idx2], source.iloc[cut_idx2:]
    if gen_file:
        df_test.to_csv(test_path, index=None, sep=',')
        df_eval.to_csv(eval_path, index=None, sep=',')
        df_train.to_csv(train_path, index=None, sep=',')
    del df_train, df_eval, df_test
    gc.collect()
"""
input_path = "./data/criteo.csv"
train_file = "./cache/train.csv"
eval_file = "./cache/eval.csv"
test_file = "./cache/test.csv"
fieldList = ['Label','I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2', 'C3',
             'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
             'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
sample(input_path, train_file, eval_file, test_file, debug=1000, col_index=fieldList)
"""


# In[3]:


def scan(file, feat_cnt):
    for row in csv.DictReader(open(file)):
        for key, val in row.items():
            if 'C' in key:
                if val == '':
                    feat_cnt[str(key) + '#' + 'absence'] += 1
                else:
                    feat_cnt[str(key) + '#' + str(val)] += 1
"""
# 统计category feat的词频
feat_cnt = defaultdict(lambda: 0)
scan(train_file, feat_cnt)
scan(eval_file, feat_cnt)
scan(test_file, feat_cnt)
print(feat_cnt)
"""


# In[4]:



"""
# 离散特征判断为长尾的阈值
T = 4 
"""
#考虑连续特征离散化和长尾特征之后，统计训练集和测试集的feat
def get_feat(featSet, file, tail=4):
    for row in csv.DictReader(open(file)):
        for key, val in row.items():
            if 'I' in key and key != "Id":
                if val == '':
                    featSet.add(str(key) + '#' + 'absence')
                else:
                    val = int(float(val))
                    if val > 2:
                        val = int(math.log(float(val)) ** 2) # 离散化
                    else:
                        val = 'SP' + str(val)
                    featSet.add(str(key) + '#' + str(val))
                continue
            if 'C' in key:
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    feat = str(key) + '#' + str(val)
                if feat_cnt[feat] > tail:
                    featSet.add(feat)
                else:
                    featSet.add(str(key) + '#' + str(feat_cnt[feat])) # 长尾数据不计value,只计频次
                continue
"""
T = 4
featSet = set()
get_feat(featSet, train_file, T)
get_feat(featSet, test_file, T)
get_feat(featSet, eval_file, T)
print('train and test data total feat num:', len(featSet))
"""


# In[5]:


def convert_to_ffm(src_path, dst_path, fieldIndex, featIndex):
    out = open(dst_path, 'w')
    for row in csv.DictReader(open(src_path)):
        feats = []
        feats.append(row['Label'])
        for key, val in row.items():
            if key == 'Label':
                continue
            if 'I' in key and key != "Id":
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    val = int(float(val))
                    if val > 2:
                        val = int(math.log(float(val)) ** 2)
                    else:
                        val = 'SP' + str(val)
                    feat = str(key) + '#' + str(val)
                feats.append(str(fieldIndex[key]) + ':' + str(featIndex[feat]) + ':1')
                continue
            if 'C' in key:
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    feat = str(key) + '#' + str(val)
                if feat_cnt[feat] > T:
                    feat = feat
                else:
                    feat = str(key) + '#' + str(feat_cnt[feat])
                feats.append(str(fieldIndex[key]) + ':' + str(featIndex[feat]) + ':1')
                continue
        out.write(' '.join(feats) + '\n')
    out.close()

if __name__ == '__main__':
    input_path = "../data/criteo.csv"
    train_file = "../cache/train.csv"
    eval_file = "../cache/eval.csv"
    test_file = "../cache/test.csv"
    out_dir = "../data/sample_1000w"
    fieldList = ['Label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1',
                 'C2', 'C3',
                 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
                 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    sample(input_path, train_file, eval_file, test_file, cols=fieldList, debug=10000000)
    print('sample done.')

    # 统计category feat词频
    feat_cnt = defaultdict(lambda: 0)
    scan(train_file, feat_cnt)
    scan(eval_file, feat_cnt)
    scan(test_file, feat_cnt)
    T = 1 # category feat长尾阈值
    featSet = set()
    # continuous feat离散化  category feat去长尾
    get_feat(featSet, train_file, T)
    get_feat(featSet, eval_file, T)
    get_feat(featSet, test_file, T)
    print('train and test data total feat num:', len(featSet))

    # 特征值编号
    featIndex = dict()
    for index, feat in enumerate(featSet, start=1):
        featIndex[feat] = index
        # print(index, feat)
    print('feat dict num:', len(featIndex))

    # 特征名（域）编号
    fieldIndex = dict()
    for index, field in enumerate(fieldList, start=0):
        fieldIndex[field] = index
    print('field dict num:', len(fieldIndex))
    
    convert_to_ffm(train_file, "{}/train.ffm".format(out_dir), fieldIndex, featIndex)
    convert_to_ffm(eval_file, "{}/eval.ffm".format(out_dir), fieldIndex, featIndex)
    convert_to_ffm(test_file, "{}/test.ffm".format(out_dir), fieldIndex, featIndex)
    with open("{}/summry".format(out_dir), 'w') as sm:
        sm.write("num_feature_values : {} \n \
                  num_field_values: {}".format(len(featIndex),len(fieldIndex)))
