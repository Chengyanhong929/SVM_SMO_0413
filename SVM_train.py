from __future__ import division
#导入python未来支持的语言特征division,精确除法
import sys
import math
import numpy as np
import random
import copy
import time#计算训练时间
import torch

#访问当前版本号
if sys.version_info[0] >= 3:
    xrange = range

#计算核值
def calcuKernelValue(train_x, sample_x, kernelOpt):
    #核的类型、参数、样本数量、核值
    kernelType = kernelOpt[0]
    kernelPara = kernelOpt[1]
    numSamples = np.shape(train_x)[0]
    kernelValue = np.mat(np.zeros((numSamples, 1)))

    if kernelType == "linear":
        kernelValue = train_x * sample_x.T
    elif kernelOpt[0] == "rbf":#表示算法使用高斯核函数
        sigma = kernelPara#sigma等于核的参数
        for i in xrange(numSamples):
            diff = train_x[i, :] - sample_x
            kernelValue[i] = math.exp(diff * diff.T / (-2 * sigma ** 2))
    else:
        print("The kernel is not supported")
    return kernelValue


#核函数求内积
#计算给定训练集和核类型的核矩阵
def calcKernelMat(train_x, kernelOpt):
    numSamples = np.shape(train_x)[0]
    kernealMat = np.mat(np.zeros((numSamples, numSamples)))
    for i in xrange(numSamples):
        kernealMat[:, i] = calcuKernelValue(train_x, train_x[i], kernelOpt)
    return kernealMat

# SVM参数
#定义一个仅仅用于存储变量和数据的结构
class svmSruct(object):

    def __init__(self, trainX, trainY, c, tolerance, maxIteration, kernelOption):
        self.train_x = trainX  #每行代表一个样本
        self.train_y = trainY  #响应的标签
        self.C = c             #松弛变量
        self.toler = tolerance #迭代的终止条件
        self.maxIter = maxIteration  #最大迭代次数
        self.numSamples = np.shape(trainX)[0]  #样本数量
        self.alphas = np.mat(np.zeros((self.numSamples, 1))) #所有样本的拉格朗日因子
        self.b = 0
        self.errorCache = np.mat(np.zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMat(self.train_x, self.kernelOpt)

#计算alpha_i的误差
def calcError(svm, alpha_i):
    func_i = np.multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_i] + svm.b
    erro_i = func_i - svm.train_y[alpha_i]
    return erro_i
#优化alpha_j后更新alpha_j的错误缓存
def updateError(svm, alpha_j):
    error = calcError(svm, alpha_j)
    svm.errorCache[alpha_j] = [1, error]

# 选取一对 alpha_i 和 alpha_j，使用启发式方法
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]#标记为有效（已优化）
    alpha_index = np.nonzero(svm.errorCache[:, 0])[0]#小地毯返回数组
    maxstep = float("-inf")
    alpha_j, error_j = 0, 0

    #用最大迭代步长求alpha
    if len(alpha_index) > 1:
        # 遍历选择最大化 |error_i - error_j| 的 alpha_j
        for alpha_k in alpha_index:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_i - error_k) > maxstep:
                maxstep = abs(error_i - error_k)
                alpha_j = alpha_k
                error_j = error_k
    #如果第一次进入这个循环，则随机选择alpha j
    else:
        # 最后一个样本，与之配对的 alpha_j采用随机选择
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = random.randint(0, svm.numSamples - 1)
        error_j = calcError(svm, alpha_j)
    return alpha_j, error_j

# 内循环
# 优化alpha i和alpha j的内环
def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)
    error_i_ago= copy.deepcopy(error_i)
    ###检查并找到违反KKT条件的阿尔法
    ##满足KKT条件
    # 1）yi*f（i）>=1和alpha==0（边界外）
    # 2）yi*f（i）=1和0<alpha<C（在边界上）
    # 3）yi*f（i）<=1和alpha==C（在边界之间）
    ##违反KKT条件
    # 因为y[i]*E_i=y[i]*f（i）-y[i]^2=y[i]*f（i）-1，所以
    # 1）如果y[i]*E_i<0，那么yi*f（i）<1，如果alpha<C，违反！（阿尔法=C将是正确的）
    # 2）如果y[i]*E_i>0，那么yi*f（i）>1，如果alpha>0，则违反！（alpha=0将是正确的
    # 3）如果y[i]*E_i=0，那么yi*f（i）=1，它在边界上，不需要优化


    if (svm.train_y[alpha_i] * error_i < -svm.toler and svm.alphas[alpha_i] < svm.C) or (svm.train_y[alpha_i] * error_i > svm.toler and svm.alphas[alpha_i] > 0):
        # 第一步：选择aplha_j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_ago = copy.deepcopy(svm.alphas[alpha_i])
        alpha_j_ago = copy.deepcopy(svm.alphas[alpha_j])# 保存更新前的aplpha值，使用深拷贝
        error_j_ago = copy.deepcopy(error_j)
        #第二步：计算alpha j的边界L和H，即最大最小
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]: # 如果yi和yj的标签不一样
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i]) # alphas[j]new的取值范围
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:# 如果yi和yj的标签一样
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)# alphas[j]new的取值范围
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            #print("L==H");
            return 0
        #计算eta，样本i和j的相似性
        #计算eta = -2 * Kij + Kii + Kjj，而这儿eta = 2 * Kij - Kii - Kjj, 所以下面公式中用的是减号
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] - svm.kernelMat[alpha_j, alpha_j]

        # 更新aplha_j
        svm.alphas[alpha_j] = alpha_j_ago - svm.train_y[alpha_j] * (error_i - error_j) / eta

        # 用于调整aj值，让aj在H和L的范围内
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        elif svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # 问题：为什么只判断alpha_j?
        #如果alpha j不动，就返回
        if (abs(alpha_j_ago - svm.alphas[alpha_j]) < 0.00001):
            # print("j not moving enough");
            return 0

        #优化aipha j后更新alpha i
        svm.alphas[alpha_i] = alpha_i_ago + svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_ago - svm.alphas[alpha_j])

        # 更新阈值 b
        b1 = svm.b - error_i_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_i] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_j] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_j, alpha_j]
        #更新b
        if (svm.alphas[alpha_i] > 0) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (svm.alphas[alpha_j] > 0) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0
        #优化alpha i、j和b后更新alpha i、j的错误缓存
        # 更新 b 之后再更新误差
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0



#训练SVM
def trainSVM(train_x, train_y, c, toler, maxIter, kernelOpt):

    #train_x:训练集数据
    #train_y:训练集标签
    #c:常数
    #toler:容错率
    #maxIter:最大循环次数
    train_start = time.time()    #训练时间开始计算
    #开始训练
    svm = svmSruct(train_x, train_y, c, toler, maxIter, kernelOpt)#调用svmSruct结构
    entire = True
    #print(type(train_x))
    alphaPairsChanged = 0#用来记录alpha是否已经进行优化
    iter = 0   # 初始化迭代次数

    # 迭代终止条件：
    # 条件1：达到最大迭代次数
    # 条件2：通过所有样本后，α值没有变化，
    # 换句话说，所有阿尔法（样本）都符合KKT条件

    # 迭代次数大于最大迭代次数时，退出迭代
    #遍历整个数据集都alpha也没有更新或者超过最大迭代次数, 则退出循环
    while (iter < svm.maxIter) and ((alphaPairsChanged > 0) or entire):
        alphaPairsChanged = 0
        #在所有训练样本中更新alphas
        if entire:# 遍历整个数据集#首先进行完整遍历，过程和简化版的SMO一样
            for i in xrange(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)#使用优化的SMO算法，如果oS.alphas[j]和oS.alphas[i]更新，则返回1
            # print("\tIter = %d, entire set, alpha pairs changed = %d" % (iter, alphaPairsChanged))
            iter += 1# 循环完一次，迭代次数加1
        #在alpha不为0和C（不在边界上）的示例上更新alpha
        else:
            ## numpy.nonzeros返回非零元素的下标位置
            nonBound_index = np.nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBound_index:# 遍历非边界元素
                alphaPairsChanged += innerLoop(svm, i)
                print("\tIter = %d, non boundary, alpha pairs changed = %d" % (iter, alphaPairsChanged))
            iter += 1
         #在所有样本和非边界样本上交替循环
        if entire:
            entire = False
        elif alphaPairsChanged == 0:
            entire = True
        # print("iteration number: %d" % iter)
    train_end = time.time()#训练结束时间
    print("\tnumVector VS numSamples == %d -- %d" % (len(np.nonzero(svm.alphas.A > 0)[0]), svm.numSamples))
    #输出训练用时
    print("\tTraining complete! ---------------- %fs" % (train_end - train_start))
    return svm
#测试函数
def testSVM(svm_test_Classifier,test_sets,test_labels):
    num_test_Samples=np.shape(test_sets)[0]
    #np.nonzero是用于得到数组array中非零元素的位置(数组索引)的函数
    support_vector_index=np.nonzero(svm_test_Classifier.alphas.A > 0)[0]
    support_vector=svm_test_Classifier.train_x[support_vector_index]
    suppport_vector_label=svm_test_Classifier.train_y[support_vector_index]
    suppport_vector_alphas=svm_test_Classifier.alphas[support_vector_index]

    #匹配数量统计
    match_Counts=0
    match_good_count=0
    match_bad_count=0
    #预测标签
    predict_label = []

    for i in range(num_test_Samples):
        kelnel_value=calcuKernelValue(support_vector,test_sets[i, : ],svm_test_Classifier.kernelOpt)
        #预测
        predict=kelnel_value.T*np.multiply(suppport_vector_label,suppport_vector_alphas)+svm_test_Classifier.b
        #预测标签
        predict_label.append(str(np.sign(predict)))
        if np.sign(predict)==np.sign(test_labels[i]):
            match_Counts+=1
            if np.sign(test_labels[i])==1:
                match_good_count+=1
                #print("good[i] VS ")
            else:
                match_bad_count+=1
            # if np.sign(test_labels[i])==1:
            #     match_good_count+=1
            # else:
            #     match_bad_count+=1
    print("\tnumRight VS numTest == %d -- %d" % (match_Counts, num_test_Samples))
    print("\t分类正确的测试集中，goodNum VS badNum== %d -- %d" %(match_good_count,match_bad_count))
    accuracy=float(match_Counts)/num_test_Samples
    return accuracy,predict_label,match_good_count,match_bad_count