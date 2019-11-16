from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import style
import numpy as np
'''设置绘图风格'''
style.use('ggplot')
'''生成演示用样本数据'''
data1 = np.random.normal(0,0.3,(1000,2))
data2 = np.random.normal(1,0.2,(1000,2))
data3 = np.random.normal(2,0.3,(1000,2))

data = np.concatenate((data1,data2,data3))

# data_tsne = TSNE(learning_rate=100).fit_transform(data)
'''搭建Mean-Shift聚类器'''
clf=MeanShift()
'''对样本数据进行聚类'''
predicted=clf.fit_predict(data)
colors = [['red','green','blue','grey'][i] for i in predicted]
'''绘制聚类图'''
plt.scatter(data[:,0],data[:,1],c=colors,s=10)
plt.title('Mean Shift')
plt.show()