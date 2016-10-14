#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'zsy'
__mtime__ = '2016/10/13'"""
import csv
import math
import os
import pickle
from datetime import datetime
from queue import Empty

import numpy as np
import pynlpir
from scipy.sparse import csr_matrix
from sklearn.preprocessing.data import normalize

BasePath = os.path.dirname(__file__)

TRAINSETFILE = os.path.join(BasePath, 'data/user_tag_query.2W.TRAIN')
TESTSETFILE = os.path.join(BasePath, 'data/user_tag_query.2W.TEST')
TEMPFILE = os.path.join(BasePath, 'temp/dataset')
RESULTFILE = os.path.join(BasePath, 'data/result.csv')


class Reader():
    def __init__(self):
        # 区分训练集和测试集，是否要分词
        # 训练集和测试集的区别只在于训练集的前四项为用户属性
        # 而测试集的前1项为用户属性
        # 如果分词，读取后的类里面包含的有用信息是：
        # 用户信息列表
        # 用户词频列表
        # 总词典
        self.userlist = []
        self.userinfo = []
        self.dict = {}

    def init(self, filename=TRAINSETFILE, IsTraining=True, IsSegment=True):
        with open(filename, encoding='GB18030') as file:
            filereader = csv.reader(file, dialect='excel-tab', quoting=csv.QUOTE_NONE)
            if not IsSegment:
                for item in filereader:
                    self.userlist.append(item)
            else:
                pynlpir.open()
                if IsTraining:
                    infoflag = 4
                else:
                    infoflag = 1
                # count_test =0
                for userquery in filereader:
                    userdict = {}
                    self.userinfo.append(userquery[:infoflag])
                    for item in userquery[infoflag:]:
                        for word in pynlpir.segment(item, pos_tagging=False):
                            if word not in self.dict.keys():
                                self.dict[word] = 0
                            if word in userdict.keys():
                                userdict[word] += 1
                            else:
                                userdict[word] = 1
                    self.userlist.append(userdict)
                    # count_test +=1
                    # if count_test>100:
                    #   break
                pynlpir.close()
        self.IsTraining = IsTraining
        self.IsSegment = IsSegment
        self.IsDF = False

    def segment(self):
        # 如果没有分词的话，对用户词典进行分词操作
        if self.IsSegment:
            pass
        else:
            pynlpir.open()
            for i, userquery in enumerate(self.userlist):
                if self.IsTraining:
                    infoflag = 4
                else:
                    infoflag = 1
                userdict = {}
                self.userinfo.append(userquery[:infoflag])
                for item in userquery[infoflag:]:
                    for word in pynlpir.segment(item, pos_tagging=False):
                        if word not in self.dict.keys():  # todo changed
                            self.dict[word] = 0  # todo changed
                        if word in userdict.keys():
                            userdict[word] += 1  # todo changed
                        else:
                            userdict[word] = 1  # todo changed
                self.userlist[i] = userdict
            pynlpir.close()
            self.IsSegment = True

    def df(self):
        # 计算df
        if not self.IsSegment:
            self.segment()
        for key in self.dict.keys():
            for userdict in self.userlist:
                if key in userdict:
                    self.dict[key] += 1
        self.IsDF = True

    def tf_idf(self):
        # 计算tf——idf
        if not self.IsDF:
            self.df()
        N = len(self.userlist)
        for i, userdict in enumerate(self.userlist):
            for key, value in userdict.items():
                self.userlist[i][key] = (math.log(value, 10) + 1) * math.log(N / self.dict[key], 10)  # todo changed

    def save(self, filename):
        # 把用户词典写入csv文档中
        with open(filename, 'w', encoding='GB18030') as file:
            for i, user in enumerate(self.userinfo):
                file.write('%s,%s,%s,%s' % (user[0], user[1], user[2], user[3]))
                for item in self.userlist[i].items():
                    file.write(',(%s %s)' % (item[0], item[1]))
                file.write('\n')


def dump(filename, obj):
    # 把用户信息和词频向量保存在temp中，以dump的方式
    with open(filename + '_pickle', 'wb') as file:
        pickle.dump(obj, file)


def cos_similar(dict_a, dict_b):
    # 计算余弦相似度
    sum_a = 0
    sum_b = 0
    similar = 0
    for a in dict_a.keys():
        sum_a += dict_a[a] ** 2
        if a in dict_b.keys():
            similar += dict_a[a] * dict_b[a]
    for b in dict_b.values():
        sum_b += b * b
    return similar / math.sqrt(sum_b * sum_a)


def similar(dict_a, dict_list):
    # 返回相似度降序排序的索引，
    similarlist = [cos_similar(dict_a, dict_b) for dict_b in dict_list]
    return list(np.argsort(similarlist))[::-1]


def similar_mat(list_a, list_b):
    # 计算用户间的相似度矩阵
    sim_mat = []
    for dict_a in list_a:
        begin = datetime.now()
        sim_mat.append([cos_similar(dict_a, dict_b) for dict_b in list_b])
        end = datetime.now()
        print(end - begin)
    return sim_mat


def similar2knn(csrcosine, k):
    # 利用相似度矩阵得到前k个最匹配用户的序号
    knn_mat = []
    for i in range(csrcosine.shape[0]):
        vector = csrcosine[i].toarray()  # 这里vector是一个1*20000的matrix
        indicesarray = vector.argsort()[::-1][0][:k]
        knn_mat.append(indicesarray)
    return knn_mat


def knn(list_a, list_b, k):
    # 直接得到前k个最匹配用户的序号
    knn_mat = []
    for a in list_a:
        knn_mat.append(similar(a, list_b)[:k])
    return knn_mat


def key_max(dict):
    return max(dict.items(), key=lambda x: x[1])[0]


def inference(testinfo, traininfo, knn_mat, filename=RESULTFILE):
    # 利用两个userinfo列表里面的信息，以及a对b的匹配信息，推断a的属性
    # 并写入csv文件中
    result = []
    with open(filename, 'w') as file:
        for i, user in enumerate(testinfo):
            # 匹配用户信息不确定的不进行匹配
            AGE = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
            GEN = {'1': 0, '2': 0}
            EDU = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
            for j in knn_mat[i]:  # test向量中的一项
                user_b = traininfo[j]
                if user_b[1] in AGE.keys():
                    AGE[user_b[1]] += 1
                if user_b[2] in GEN.keys():
                    GEN[user_b[2]] += 1
                if user_b[3] in EDU.keys():
                    EDU[user_b[3]] += 1
            user.append(key_max(AGE))
            user.append(key_max(GEN))
            user.append(key_max(EDU))
            file.write('%s %s %s %s\n' % (user[0], user[1], user[2], user[3]))
            result.append(user)
    return result


def similar_mat_one(testitem, list_b):
    # 计算用户间的相似度矩阵
    begin = datetime.now()
    sim_vector = [cos_similar(testitem, dict_b) for dict_b in list_b]
    end = datetime.now()
    print(end - begin)
    return sim_vector


def simuser(testqueue, userlist, lock, sim_mat):
    while True:
        try:
            itemindex, testitem = testqueue.get(block=False)  # 默认是阻塞的
        except Empty:
            print('empty')
            return
        sim_vector = similar_mat_one(testitem, userlist)
        lock.acquire()
        sim_mat.append([itemindex, sim_vector])
        lock.release()


def inputQ(queue, test_userlist):
    for i, item in enumerate(test_userlist):
        queue.put((i, item))


def parsetrain2sparse(train, dictarray):
    row = []
    col = []
    data = []
    for i, token in enumerate(dictarray):  # row
        for j, userdict in enumerate(train.userlist):  # column
            if token in userdict:
                row.append(i)
                col.append(j)
                data.append(userdict[token])
    return csr_matrix((data, (row, col)), shape=(len(dictarray), len(train.userlist)))


def parsetest2sparse(test, dictarray):
    row = []
    col = []
    data = []
    for j, token in enumerate(dictarray):  # column
        for i, userdict in enumerate(test.userlist):  # row
            if token in userdict:
                row.append(i)
                col.append(j)
                data.append(userdict[token])
    return csr_matrix((data, (row, col)), shape=(len(test.userlist), len(dictarray)))


if __name__ == '__main__':
    begin = datetime.now()
    print(begin)
    train = Reader()  # 读train数据
    train.init()
    # train.save('train')
    step1 = datetime.now()
    print(step1)
    test = Reader()  # 读test数据
    test.init(filename=TESTSETFILE, IsTraining=False, IsSegment=True)
    # test.save('test')
    step2 = datetime.now()
    print(step2)
    train.tf_idf()  # 计算train的tf_idf
    test.tf_idf()  # 计算test的tf_idf
    # train.save('train_tf_idf')
    # test.save('test_tf_idf')
    step3 = datetime.now()
    print(step3)

    dump('train', train)
    dump('test', test)

    ''' ------------------------------华丽的json分割线-----------------------------------'''
    # 使用这里的代码可替代上面的代码，直接读取上面的分词结果
    # import json
    #
    # train = Reader()
    # with open('train_userlist') as f:
    #     train.userlist = json.load(f)
    # with open('train_userinfo') as f:
    #     train.userinfo = json.load(f)
    # with open('train_dict') as f:
    #     train.dict = json.load(f)
    # test = Reader()
    # with open('test_userlist') as f:
    #     test.userlist = json.load(f)
    # with open('test_userinfo') as f:
    #     test.userinfo = json.load(f)
    # with open('test_dict') as f:
    #     test.dict = json.load(f)
    # print('read json file finished')

    ''' ------------------------------华丽的json分割线-----------------------------------'''

    dictlist = list(train.dict.keys())
    csrtrain = parsetrain2sparse(train, dictlist)
    print('parsetrain2sparse finish %s' % datetime.now())
    csrtest = parsetest2sparse(test, dictlist)
    print('parsetest2sparse finish %s' % datetime.now())
    dump('csrtrain', csrtrain)
    dump('csrtest', csrtest)

    ''' ------------------------------华丽的sparse分割线-----------------------------------'''

    # 使用这里的代码替代上面的代码，直接读取sparse矩阵
    # with open('csrtrain_pickle', 'rb') as f: # float16
    #     csrtrain = pickle.load(f)
    # with open('csrtest_pickle', 'rb') as f:
    #     csrtest = pickle.load(f)
    # csrtrain= csrtrain.astype(np.float64)

    ''' ------------------------------华丽的sparse分割线-----------------------------------'''
    csctrainnor = normalize(csrtrain, norm='l2', axis=0)
    csrtrain = csctrainnor.tocsr()
    csrtrain = csrtrain.astype(np.float16)

    csrcosine = csrtest.dot(csrtrain)  # 矩阵乘法
    print('dot finish %s' % datetime.now())
    # sim_mat = [list(row) for row in csrcosine.toarray()]  # 该代码占用内存太大，直接死机

    # sim_mat = similar_mat(test.userlist, train.userlist)  # 计算匹配度矩阵
    # with open(os.path.join(BasePath, 'temp/sim_mat'), 'wb') as file:  # 将匹配矩阵存起来，因为该数据很重要
    #     pickle.dump(sim_mat, file)
    print('start knn')
    knn_mat = similar2knn(csrcosine, 10)  # 通过匹配矩阵来计算匹配用户
    step4 = datetime.now()
    print(step4)
    result = inference(test.userinfo, train.userinfo, knn_mat)  # 对用户进行推断并写入
    end = datetime.now()
    print(end)
