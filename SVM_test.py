from __future__ import division
#导入python未来支持的语言特征division,精确除法
from numpy import *
from SVM_train import *
import torch
from sklearn.model_selection import train_test_split

import sys
if sys.version_info[0] >= 3:
    xrange = range


file_open_txt=open('D:\\LenovoQMDownload\\PythonProject\\SVM_SMO\\data_Split\\2d_400_data.txt')
def test_svm_hanshu():
    data_Set=[]
    label_Set=[]
    for line in file_open_txt.readlines():
        lineArr=line.strip().split(',')
        data_Set.append([float(data) for data in lineArr[:-2]])
        if int(lineArr[-2]) == 1:
            label_Set.append(1)#good=1
        else:
            label_Set.append(-1)#bad=2
        # data_Set.append([float(data) for data in lineArr[:-1]])
        # if int(lineArr[-1]) == 1:
        #     label_Set.append(1)  # good=1
        # else:
        #     label_Set.append(2)  # bad=2

    data_Set=np.mat(data_Set)
    label_Set=np.mat(label_Set).T

    print("Step 1:Loading data....................................")
    train_set=data_Set[0:280,:]
    train_label=label_Set[0:280,:]
    test_set=data_Set[280:400,:]
    test_label=label_Set[280:400,:]
    # train_set=data_Set[0:100,:]
    # train_label=label_Set[0:100,:]
    # test_set=data_Set[101:129,:]
    # test_label=label_Set[101:129,:]
    #train_x, test_x, train_y, test_y = train_test_split(data_Set, label_Set, test_size=0.3)

    print("Step 2:Training........................................")
    c=10#C越大表明离群点对目标函数影响越大，也就是越不希望看到离群点
    toler=0.001
    maxIter=20
    kernelOption=("rbf",100)
    #kernelOption = ("linear", 0)
    svm_Classification=trainSVM(train_set,train_label,c,toler,maxIter,kernelOption)

    print("Step 3:Testing..........................................")
    accuracy,predict_label,match_good_count,match_bad_count=testSVM(svm_Classification,test_set,test_label)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^next is result^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    print("\tThe classify accuracy is: %f" %(accuracy*100))

test_svm_hanshu()
print("-----------------------------------------------------------------------------")
