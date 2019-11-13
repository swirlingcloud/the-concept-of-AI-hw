import numpy as np


List=np.random.uniform(0,1,size=[500000,200])#随机生成500000个200维的点

# achieve the function of PCA
def pca(X):#k is the components you want
  #mean of each feature
  n_features = np.shape(X)
  mean=np.mean(X,axis=0) 
  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)

  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [[np.abs(eig_val[i]), eig_vec[:,i]] for i in range(n_features[1])]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top n to get the k compents
  a=np.sum(eig_pairs,axis=0)
  b=0#the sum of front k 
  n=0

  eig=list(eig_pairs)
  advip=[]
  for ele in eig:
      advip.append(ele[0])
  while b<0.9*a[0]:#keep 90% energy
      b=b+advip[n]
      n=n+1
  
  print("最终得到的维度是：",n)
   
  feature=np.array([ele[1] for ele in eig_pairs[:n]])
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  return data

donelist=pca(List)


