'''
#using svm to predict
# preprocesing:
# using chi_square to selecte 5000 relevant feature
# using SVD to decomposition the feature to 100 dimantion
# predictor
'''

from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.externals import joblib

import numpy as np

import pickle
import os

BasePath = os.path.dirname(os.path.dirname(__file__))

trainfile = os.path.join(BasePath,'temp\csrtrain_pickle')
testfile = os.path.join(BasePath,'temp\csrtest_pickle')
labelfile = os.path.join(BasePath,'temp/TRAIN_info')
test_info = os.path.join(BasePath,'temp/test_info')
resultfile = os.path.join(BasePath,'data/result2.csv')

################################################################
######调参看这里########

#特征选择的参数

selector = [
            SelectKBest(chi2,k=5000),
            SelectKBest(chi2,k=5000),
            SelectKBest(chi2,k=5000)
            ]




#支持向量机的参数
clf= [
        SVC(),
        SVC(),
        SVC(),
        ]
###############################################################

def getdata(filename = trainfile):
    with open(filename,'rb') as file:
        data = pickle.load(file)
    data.format = 'csr'
    data = data.transpose().tocsr()
    return data

def getlabel(filename = labelfile):
    with open(filename,'rb') as file:
        label = np.array(pickle.load(file))
    age_label = label[:,1]
    gender_label = label[:,2]
    edu_label = label[:,3]
    return age_label,gender_label,edu_label



# def train(data,label):
#     clf = RandomForestClassifier(n_estimators= 50,n_jobs =-1)
#     clf.fit(data,label)
#     return clf


if __name__=='__main__':
    #label
    age,gender,edu = getlabel()
    print(1)
    #load the data
    data = getdata()

    age = np.array(age)
    gender = np.array(gender)
    edu = np.array(edu)

    label = [age,gender,edu]

    #ignore the unlabeled data
    aid = np.array([False if i =='0' else True for i in age]) 
    gid = np.array([False if i =='0' else True for i in gender]) 
    eid = np.array([False if i =='0' else True for i in edu])

    lid = [aid,gid,eid]




    #load test data
    with open(testfile,'rb') as file:
        testdata = pickle.load(file)
        testdata.format ='csr' 

    result = []
    cv_score = []


    for  i in range(len(label)):
        mdata = data[lid[i]]
        mtest = testdata[:]
        mlabel = label[i][lid[i]]

        # #using chi_square to select feature in train and test
        # mdata = selector[i].fit_transform(mdata,mlabel)
        # mtest = selector[i].transform(mtest)


        #using svd to decomposition the train and test data
        svd = TruncatedSVD(100)
        mdata = svd.fit_transform(mdata)
        mtest = svd.transform(mtest)

        # if we want cross validation
        # clf 是要调节的参数之一
        cv_score.append(cross_val_score(clf[i],mdata,mlabel,cv=5))


        ###################################################################
        #下面的代码在CV的时候不需要
        # #using svm to classify the test
        # clf[i].fit(mdata,mlabel)

        # mpredict = clf[i].predict(mtest)

        # result.append(mpredict)

    score = np.mean([np.mean(cv) for cv in cv_score])
    print(U'预期的得分是')
    print(score)
    print(U'每一项的得分是')
    for cv in cv_score:
        print(cv)
        

###############################################################################################
    #下面的代码是预测用的，cv的时候没用
    # #write the result to file
    # with open(test_info,'rb') as file:
    #     test = pickle.load(file)
    # with open(resultfile,'w') as file:
    #     for i in range(len(test)):
    #         file.write(test[i][0]+' '+result[0][i]+' '+result[1][i]+' '+result[2][i]+'\n')








###################################################################################################
####### ugly code ##################
    # #get different copy of data
    # data1 = data[aid]
    # data2 = data[gid]
    # data3 = data[eid]

    # #get a copy off test data, then the test data will not be changed in prediction
    # test1 = testdata[:]
    # test2 = testdata[:]
    # test3 = testdata[:]

    # # #using chi_square to selector the feature from data

    # # selector1 = SelectKBest(chi2,k=5000)
    # # selector2 = SelectKBest(chi2,k=5000)
    # # selector3 = SelectKBest(chi2,k=5000)

    # # data1 = selector1.fit_transform(data1,age[aid])
    # # data2 = selector2.fit_transform(data2,gender[gid])
    # # data3 = selector3.fit_transform(data3,edu[eid])

    # # # using chi_square to selector feature
    # # test1 = selector1.transform(test1)
    # # test2 = selector2.transform(test2)
    # # test3 = selector3.transform(test3)


    # #training svd and decomposition
    # svd1 = TruncatedSVD(100)
    # svd2 = TruncatedSVD(100)
    # svd3 = TruncatedSVD(100)
    
    # data1 = svd1.fit_transform(data1)
    # data2 = svd2.fit_transform(data2)
    # data3 = svd3.fit_transform(data3)


    # # using svd to decomposition
    # test1 = svd1.transform(test1)
    # test2 = svd1.transform(test2)
    # test3 = svd1.transform(test3)


    # # # cross validation
    # # clf = SVC()

    # # score_test1 = cross_val_score(clf,data1,age[aid],cv=2)
    # # score_test2 = cross_val_score(clf,data2,gender[gid],cv=5)
    # # score_test3 = cross_val_score(clf,data2,edu[eid],cv=5)


    # # training and prediction

    # clf1 = SVC()  #to add some parameters
    # clf2 = SVC()   #
    # clf3 = SVC()  #also need to add some parameters

    # clf1.fit(data1,age[aid])
    # clf2.fit(data2,gender[gid])
    # clf3.fit(data3,edu[eid])


    # agep = clf1.predict(testdata)
    # genderp = clf2.predict(testdata)
    # edup = clf3.predict(testdata)


    # #write the result to file
    # with open(test_info,'rb') as file:
    #     test = pickle.load(file)
    # with open(resultfile,'w') as file:
    #     for i in range(len(test)):
    #         file.write(test[i][0]+' '+agep[i]+' '+gender[i]+' '+edu[i]+'\n')





############################################################################################
##########  test  and find solution ##########

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

    