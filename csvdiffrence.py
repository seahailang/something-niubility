#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'zsy'
__mtime__ = '2016/10/13'"""
import csv
import os

BasePath = os.path.dirname(__file__)
RESULTFILE = os.path.join(BasePath, 'data/result.csv')
RESULTPREFILE = os.path.join(BasePath, 'data/result_pre.csv')

if __name__ == '__main__':
    respre=[]
    keyword = 0  # 关键词数量
    itemCount = 0  # 记录数量
    ageList = [0] * 7
    genderList = [0] * 3
    educationList = [0] * 7
    with open(RESULTPREFILE, encoding='gbk') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for item in reader:
                respre.append(item)
                itemCount += 1
                keyword += len(item) - 4
                ageList[int(item[1])] += 1
                genderList[int(item[2])] += 1
                educationList[int(item[3])] += 1
    print(itemCount)
    print(keyword)
    print(ageList)
    print(genderList)
    print(educationList)

    res=[]
    keyword = 0  # 关键词数量
    itemCount = 0  # 记录数量
    ageList = [0] * 7
    genderList = [0] * 3
    educationList = [0] * 7
    with open(RESULTFILE, encoding='gbk') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for item in reader:
                res.append(item)
                itemCount += 1
                keyword += len(item) - 4
                ageList[int(item[1])] += 1
                genderList[int(item[2])] += 1
                educationList[int(item[3])] += 1
    print(itemCount)
    print(keyword)
    print(ageList)
    print(genderList)
    print(educationList)

    TRAINSETFILE = 'user_tag_query.2W.TRAIN'
    keyword = 0  # 关键词数量
    itemCount = 0  # 记录数量
    ageList = [0] * 7
    genderList = [0] * 3
    educationList = [0] * 7
    ageDict = {'unknown': 0, '0-18': 0, '19-23': 0, '24-30': 0, '31-40': 0, '41-50': 0, '51-999': 0}  # 0-6共7类
    genderDict = {'unknown': 0, 'male': 0, 'female': 0}  # 0-2共3类
    educationDict = {'unknown': 0, 'phd': 0, 'master': 0, 'undergraduate': 0, 'senior': 0, 'junior': 0,
                     'primary': 0}  # 0-6共7类

    with open(TRAINSETFILE, encoding='GB18030') as csvTrainfile:
        reader = csv.reader(csvTrainfile, dialect='excel-tab', quoting=csv.QUOTE_NONE)
        for item in reader:
            itemCount += 1
            keyword += len(item) - 4
            ageList[int(item[1])] += 1
            genderList[int(item[2])] += 1
            educationList[int(item[3])] += 1

    print(itemCount)
    print(keyword)
    print(ageList)
    print(genderList)
    print(educationList)

    dif = 0
    for i,itempre in enumerate(respre):
        item = res[i]
        for j, val in enumerate(itempre):
            if item[j] != val:
                dif += 1
    print('不同个数：%d' % dif)
    print('差异率：%f （百分比）' % (dif*100/60000))
