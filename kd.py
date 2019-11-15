import numpy as np
from sklearn.neighbors import KDTree
from time import time


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, createCent = randCent):
    m = np.shape(dataSet)[0]   
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True  
    while clusterChanged:
        clusterChanged = False;
        tree = KDTree(centroids, leaf_size=200)
        dist,ind=tree.query(dataSet, k=1)
        for i in range(m):
            if clusterAssment[i,0] != ind[i]:
                clusterChanged = True
            clusterAssment[i,:] = ind[i][0],dist[i][0]
            '''
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j 
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True  
            clusterAssment[i,:] = minIndex,minDist**2   '''      
        for cent in range(k): 
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]   
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)  
    return centroids, clusterAssment


if __name__ == '__main__':
    # 特征 + 类别
    datMat = np.mat(np.random.random(size=(10000,179)))
    starttime = time()
    myCentroids, clustAssing= kMeans(datMat,100)
    endtime = time()
    runtime = endtime - starttime
    print('run time is : %f s'%runtime)