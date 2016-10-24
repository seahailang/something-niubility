import sklearn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

import scipy
from scipy import sparse
import numpy as np

import pickle
import os

BasePath = os.path.dirname(os.path.dirname(__file__))

trainfile = os.path.join(BasePath,'temp\csrtrain_pickle')
testfile = os.path.join(BasePath,'temp\csrtest_pickle')
labelfile = os.path.join(BasePath,'temp/TRAIN_info')
test_info = os.path.join(BasePath,'temp/test_info')


def getsvd(filename = trainfile):
    with open(filename,'rb') as file:
        data = pickle.load(file)
    data.format = 'csr'
    data = data.transpose().tocsr()
    svd = TruncatedSVD(100)
    svd.fit(data)
    return svd,data

def getlabel(filename = labelfile):
    with open(filename,'rb') as file:
        label = np.array(pickle.load(file))
    age_label = label[:,1]
    gender_label = label[:,2]
    edu_label = label[:,3]
    return age_label,gender_label,edu_label



def train(data,label):
    clf = RandomForestClassifier(n_estimators= 50,n_jobs =-1)
    clf.fit(data,label)
    return clf


if __name__=='__main__':
    age,gender,edu = getlabel()
    svd,data = getsvd()
    data = svd.transform(data)
    #clf = RandomForestClassifier(n_estimators= 100,n_jobs =-1)
    #clf = SVC()

    #scores3 = cross_val_score(clf,data,age,cv=5)



    
    # data = svd.transform(data)
    # clf1 = train(data,age)
    # joblib.dump(clf1,'D:/CCF/models/ageclf.m')
    # clf2 = train(data,gender)
    # joblib.dump(clf2,'D:/CCF/models/genderclf.m')
    # clf3 = train(data,edu)
    # joblib.dump(clf3,'D:/CCF/models/educlf.m')
    # with open(testfile,'rb') as file:
    #     testdata = pickle.load(file)
    # testdata.format = 'csr'
    # testdata = svd.transform(testdata)
    # agep = clf1.predict(testdata)
    # genderp = clf2.predict(testdata)
    # edup = clf3.predict(testdata)
    # with open(test_info,'rb') as file:
    #     test = pickle.load(file)
    # with open('D:/CCF/data/result1.csv','w') as file:
    #     for i in range(len(test)):
    #         if genderp[i] == U'0':
    #             genderp[i] = U'1'
    #         if agep[i] ==U'0':
    #             agep[i] = U'1'
    #         if edu[i] ==U'1':
    #             edu[i] = '5'
    #         file.write(test[i][0]+' '+agep[i]+' '+gender[i]+' '+edu[i]+'\n')

    