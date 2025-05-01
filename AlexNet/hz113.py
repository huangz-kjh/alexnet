# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

def L2(vecXi,vecXj):
    '''
        计算欧氏距离

    Parameters
    ----------
    vecXi : 点坐标 向量
    vexXj : 点坐标 向量

    Returns
    -------
    两点坐标的欧式距离

    '''
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))

def kMeans(S, k, distMeans=L2):
    '''
    k均值聚类

    Parameters
    ----------
    S : 样本集，多维数组
    k : 簇个数
    distMeans : 距离量度函数，默认为欧氏距离计算函数
        DESCRIPTION. The default is L2.

    Returns
    -------
    sampleTag:一维数组，储存样本对应的簇标记
    cluterCents:一维数组，各簇的中心
    SSE：误差平方和

    '''
    m = np.shape(S)[0]#得到样本总数
    #print(m)
    sampleTag = np.zeros(m)#数组清零了
    #print(sampleTag)
    
    #随机产生k个初始簇中心
    n = np.shape(S)[1] #样本向量的特征数
    #print(n)
    #clusterCents = np.mat([[-1.93964824,2.33260803],[7.79822795,6.72621783],[10.64183154,0.20088133]])# 手动分配簇
    #随机分配簇中心
    clusterCents = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(S[:,j])
        rangeJ = float(max(S[:,j])- minJ)
        clusterCents[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
        
    sampleTagChanged = True
    SSE = 0.0
    while sampleTagChanged:
        sampleTagChanged = False
        SSE = 0.0
        
        #计算每个样本点到各个簇中心的距离
        for i in range(m):
            minD = np.inf
            minIndex = -1
            for j in range(k):
                d = distMeans(clusterCents[j,:],S[i,:])
                if d < minD:
                    minD = d
                    minIndex = j
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex
            SSE += minD*2
        #print(clusterCents)
        #plt.scatter(clusterCents[:,0].tolist(), clusterCents[:,1].tolist(), c='r',marker='^',linewidths=7)
        #plt.scatter(S[:,0],S[:,1],c=sampleTag,linewidths=np.power(sampleTag+0.5, 2))
        #plt.show()
        #print(SSE)
        
        #重新计算簇中心
        for i in range(k):
            ClustI = S[np.nonzero(sampleTag[:] == i)[0]]
            clusterCents[i,:] = np.mean(ClustI, axis=0)
    return clusterCents,sampleTag, SSE

if __name__ == '__main__':
    samples = np.loadtxt("kmeansSamples.txt")
    clusterCents, sampleTag, SSE = kMeans(samples, 3)
    plt.scatter(clusterCents[:,0].tolist(), clusterCents[:,1].tolist(), c='r',marker='^',linewidths=7)
    plt.scatter(samples[:,0],samples[:,1],c=sampleTag,linewidths=np.power(sampleTag+0.5, 2))
    plt.show()
    print(clusterCents)
    print(SSE)