import  matplotlib.pyplot as plt
x=[30,33,35,37,39,45,46,50]#横轴x的变量
y=[30,34,37,39,42,46,48,51]#y的变量

#迭代阙值
epl=0.1
loop_max=1000

aph=0.0001#学习率
theta1=1.0#初始化权重
theta0=0.5
time=0#迭代次数
sum=len(x)#总的数量
diff=0#函数值
error2=0#误差
error1=0


while time<=loop_max:
    time+=1
    for i in range(sum):

        diff=(theta0+theta1*x[i])-y[i]
        theta1=theta1-aph*diff*x[i]
        theta0=theta0-aph*diff

    #计算损失函数
    error1=0
    for j in range(sum):
        error1+=(y[j]-theta1*x[j]-theta0)**2/2  #变量只有一维的时候如何让参数多维
    if abs(error1-0)<epl: #绝对值判断
        break
    else:
        error2=error1

print("theta1:",theta1)
print("theta0:",theta0)
print("迭代次数",time)
print("误差率",error2)

#用来对新的列表进行赋值
y2=[0,0,0,0,0,0,0,0]
for i in range(sum):
    y2[i]=theta0+theta1*x[i]

print('the performance in the 47th training is', theta0+theta1*47)
print('the performance in the 55th training is', theta0+theta1*55)


plt.plot(x,y,'g*')   
plt.plot(x,y2,'r')
plt.show()

