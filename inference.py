
from reader_multi_process import *
from multiprocessing import Process

filelist =list(map(lambda x:os.path.join(BasePath,'temp/'+x),['test_info','test_dict','train_info','train_dict']))


def loaddata(filelist = filelist):
    data=[]
    for filename in filelist:
        with open(filename,'rb') as file:
            temp = pickle.load(file)
            data.append(temp)
    return data[0],data[1],data[2],data[3]

# def simi_write(list_a,list_b,filename):
#     sim_mat = similar_mat(list_a,list_b)
#     with open(filename,'wb') as file: 
#         pickle.dump(sim_mat,file)




def worker(usersa,usersb,list_a,list_b,filename,k):
    #并行计算的worker
    list_b = np.array(list_b)
    user_idx = []
    sim_mat = []
    for dict_a in list_a:
        idx = np.random.randint(0,20000,2000)
        sim_list = [cos_similar(dict_a,list_b[i]) for i in idx]
        user_idx.append(idx)
        sim_mat.append(sim_list)
    knn_mat = similar2knn(sim_mat,k)
    for i in range(len(user_idx)):
        user_idx[i] = user_idx[i][knn_mat[i]]
    result = inference(usersa,usersb,user_idx,filename)





if __name__ =='__main__':
    info_a,list_a,info_b,list_b = loaddata()
    cut = [(0,5000),(5000,10000),(10000,15000),(15000,20000)]
    # cut =[(0,5),(5,10)]
    list_cut = [list_a[c[0]:c[1]] for c in cut]
    
    #simi_write(list_a[cut[0][0]:cut[0][1]],list_b,'D:/CCF/temp/simi_mat')
    filename = os.path.join(BasePath,'data/result')
    plist=[]
    for i,c in enumerate(list_cut):
        plist.append(Process(target = worker,args = (info_a[cut[i][0]:cut[i][1]],info_b,c,list_b,filename+str(i),50)))
    for i in range(len(cut)):
        plist[i].start() 
    for i in range(len(cut)):
        plist[i].join()



