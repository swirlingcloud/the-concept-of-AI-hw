import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from time import time

def dis(a,b):#计算两个向量之间的距离，欧氏距离
    return np.linalg.norm(a-b)


n=int(input("请输入所对应的点的维度:"))

List=[]
for i in range(1000):
    data=np.random.rand(1,n)#随机生成1000个n维的点
    List.extend(data)

SSE=[]   
#随机生成质心
for k in range(1,9):
    cent=np.zeros([k,n])#初始值
    for i in range(k):
        x=np.random.randint(1000)
        cent[i]=List[x]
    #SSE误差的平方和   
    #循环计算距离，需要考虑二维数组的运用 distance.append(j)
    a=0

    while a<20:
        a=a+1
        datalist=[]#放多个中心点里所包含的数据集合
        for i in range(k):
            w=[]
            datalist.append(w)

        tree = KDTree(centroids, leaf_size=200)
        dist,ind=tree.query(dataSet, k=1)

        for i in range(1000):
            clusterchange=False
            sumdis=[]
            datal=[]
            for j in range(k):
               distance=[]
               distance.append(dis(List[i],cent[j]))#计算与新聚类中心的距离
               distance.append(j)
               sumdis.append(distance)
            sumdis.sort()#排序
            datal=sumdis[0]
            datalist[datal[1]].append(List[i])#第k个类的

        for i in range(k):
            cent[i]=np.mean(datalist[i],0)#对datalist数据集里点求mean

    #分类之后和每个点比较 
    SSE2=0
    for i in range(k):
        for j in range(len(datalist[i])):
            SSE2=SSE2+dis(datalist[i][j],cent[i])

    SSE.append(SSE2)

x=range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.ylim(200,1000)#需要根据自己的维度进行调整
plt.plot(x,SSE,'o-')
plt.show()


