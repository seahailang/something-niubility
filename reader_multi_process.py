import csv
import os
import pickle
import pynlpir
import math
import numpy as np

from datetime import datetime

BasePath = os.path.dirname(os.path.dirname(__file__))

TRAINSETFILE =os.path.join(BasePath,'data/user_tag_query.2W.TRAIN')
TESTSETFILE = os.path.join(BasePath,'data/user_tag_query.2W.TEST')
TEMPFILE = os.path.join(BasePath,'temp/dataset')
RESULTFILE = os.path.join(BasePath,'data/result.csv')
TEST = os.path.join(BasePath,'temp/test')
TRAIN = os.path.join(BasePath,'temp/TRAIN')


class Reader():
    def __init__(self, filename=TRAINSETFILE,IsTraining = True,IsSegment =True):
    	#区分训练集和测试集，是否要分词
    	#训练集和测试集的局别只在于训练集的前四项为用户属性
    	#而测试集的前1项为用户属性
    	#如果分词，读取后的类里面包含的有用信息是：
    	#用户信息列表
    	#用户词频列表
    	#总词典
        self.userlist = []
        self.userinfo = []
        self.dict = {}
        self.IsTraining = IsTraining
        self.IsSegment = IsSegment
        self.IsDF = False
        with open(filename,encoding = 'GB18030') as file:
            filereader = csv.reader(file,dialect = 'excel-tab',quoting = csv.QUOTE_NONE)
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
                    userdict={}
                    userdictflag ={}
                    self.userinfo.append(userquery[:infoflag])
                    for item in userquery[infoflag:]:
                        for word in pynlpir.segment(item,pos_tagging=False):
                            if word in userdict.keys():
                                userdict[word] += 1
                                userdictflag[word] = False
                            else:
                                userdict[word] = 1
                                userdictflag[word] = True
                            if word not in self.dict.keys():
                                self.dict[word] = 0
                            if userdictflag[word]:
                                self.dict[word] += 1
                    self.userlist.append(userdict)
                    # count_test +=1
                    # if count_test>100:
                    #    break
                pynlpir.close()
                self.IsDF = True

    def segment(self):
    	#如果没有分词的话，对用户词典进行分词操作
        if self.IsSegment:
            pass
        else:
            pynlpir.open()
            for i,userquery in enumerate(self.userlist):
                if self.IsTraining:
                    infoflag = 4
                else:
                    infoflag = 1
                userdict ={}
                self.userinfo.append(userquery[:infoflag])
                for item in userquery[infoflag:]:
                    for word in pynlpir.segment(item,pos_tagging=False):
                        if word not in self.dict.keys():
                            self.dict[word] = 0
                        if word in userdict.keys():
                            userdict[word] += 1
                        else:
                            userdict[word] = 1
                self.userlist[i] = userdict
            pynlpir.close()
            self.IsSegment = True

    def df(self):
    	#计算df
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
        for i,userdict in enumerate(self.userlist):
            for key,value in userdict.items():
                self.userlist[i][key] = (math.log(value, 10) + 1) * math.log(N / self.dict[key], 10)  # todo changed

    def normalize(self):
        for i, userdict in enumerate(self.userlist):
            norm = np.sqrt((np.array(list(userdict.values()))**2).sum())
            for key,value in userdict.items():
                self.userlist[i][key] = value/norm


    def dump(self,filename):
    	#把用户信息和词频向量保存在temp中，以dump的方式
        with open(filename+'_info','wb') as file:
            pickle.dump(self.userinfo,file)
        with open(filename+'_dict','wb') as file:
            pickle.dump(self.userlist,file)
        with open(filename+'_all_dict','wb') as file:
            pickle.dump(self.dict,file)
    def save(self,filename):
    	#把用户词典写入csv文档中
        with open(filename,'w',encoding = 'GB18030') as file:
            for i,user in enumerate(self.userinfo):
                file.write('%s,%s,%s,%s'%(user[0],user[1],user[2],user[3]))
                for item in self.userlist[i].items():
                    file.write(',(%s %s)'%(item[0],item[1]))
                file.write('\n')

def normalize(userlist):
    for i,userdict in enumerate(userlist):
        norm = np.sqrt((np.array(list(userdict.values()))**2).sum())
        for key ,value in userdict.items():
            userlist[i][key] = value/norm

def cos_similar(dict_a,dict_b):
	#计算余弦相似度
    sum_a = 0
    sum_b = 0
    similar = 0
    for a in dict_a.keys():
        sum_a += dict_a[a]**2
        if a in dict_b.keys():
            similar += dict_a[a]*dict_b[a]
    for b in dict_b.values():
        sum_b += b*b
    return similar/math.sqrt(sum_b*sum_a)

def similar(dict_a,dict_list,sample = True):
	#返回相似度降序排序的索引，
    if not sample:
        similarlist = [cos_similar(dict_a,dict_b) for dict_b in dict_list]
    else:
        similarlist = [cos_similar(dict_a,dict_b) for dict_b in np.random.choice(dict_list,2000)]
    return np.argsort(similarlist)[::-1]

def similar_mat(list_a,list_b):
	#计算用户间的相似度矩阵
    #list_a 是test
    #list_b 是train
	sim_mat = []
	for dict_a in list_a:
		# begin = datetime.now()
		sim_mat.append([cos_similar(dict_a,dict_b)for dict_b in list_b])
		# end = datetime.now()
		# print(end-begin)
	return sim_mat

# def similar_mat_multi_p(list_a,list_b,num_p = 10):
#     processlist = []
#     cut = [(0,5000),(5000,10000),(10000,15000),(15000,20000)]
#     for c in cut:
#         processlist.append(Process(target = similar_mat,args =(list_a[c[0]:c[1]],list_b)))
#     for i in range(4):
#         processlist[i].start()
#         processlist[i].join()




def similar2knn(sim_mat,k):
	#利用相似度矩阵得到前k个最匹配用户的序号
	knn_mat =[]
	for sim in sim_mat:
		knn_mat.append(np.argsort(sim)[::-1][:k])
	return knn_mat


def knn(list_a,list_b,k):
	#直接得到前k个最匹配用户的序号
    knn_mat = []
    for a in list_a:
        knn_mat.append(similar(a,list_b)[:k])
    return knn_mat

def key_max(dict):
    return max(dict.items(),key= lambda x:x[1])[0]


def inference(usersa,usersb,knn_mat,filename = RESULTFILE):
	#利用两个userinfo列表里面的信息，以及a对b的匹配信息，推断a的属性
	#并写入csv文件中
    result = []
    with open(filename,'w') as file:
        for i,user in enumerate(usersa):
        	#匹配用户信息不确定的不进行匹配
            AGE ={'1':0,'2':0,'3':0,'4':0,'5':0,'6':0}
            GEN ={'1':0,'2':0}
            EDU ={'1':0,'2':0,'3':0,'4':0,'5':0,'6':0}
            for j in knn_mat[i]:
                user_b = usersb[j]
                if user_b[1] in AGE.keys():
                    AGE[user_b[1]] += 1
                if user_b[2] in GEN.keys():
                    GEN[user_b[2]] += 1
                if user_b[3] in EDU.keys():
                    EDU[user_b[3]] += 1
            user.append(key_max(AGE))
            user.append(key_max(GEN))
            user.append(key_max(EDU))
            file.write('%s %s %s %s\n'%(user[0],user[1],user[2],user[3]))
            result.append(user)
    return result


if __name__ == '__main__' :
    begin = datetime.now()
    print(begin)
    train = Reader()        #读train数据
    step1 = datetime.now()
    print(step1)
    test = Reader(filename = TESTSETFILE,IsTraining=False,IsSegment =True)  #读test数据
    step2 = datetime.now()
    print(step2)
    train.tf_idf()  #计算train的tf_idf
    test.tf_idf()   #计算test的tf_idf
    train.dump(TRAIN)
    test.dump(TEST)
    step3 = datetime.now()
    print(step3)
    # sim_mat = similar_mat(test.userlist,train.userlist)   #计算匹配度矩阵
    # with open(os.path.join(BasePath,'temp/sim_mat'),'wb') as file:    #将匹配矩阵存起来，因为该数据很重要
    # 	pickle.dump(sim_mat,file)
    # knn_mat = similar2knn(sim_mat,10)    #通过匹配矩阵来计算匹配用户
    # step4 = datetime.now()
    # print(step4)
    # result = inference(test.userinfo,train.userinfo,knn_mat) #对用户进行推断并写入
    # end = datetime.now()
    # print(end)
    



    













    






