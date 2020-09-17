import random
import numpy as np
import matplotlib.pyplot as plt
import pylab

D=4
N = random.randint(1000,1500)
print('Задание a')
X=np.zeros((D, N),float)
print(X.shape)
print(X)

print('Задание b')
X[0]=[random.randint(25,210) for i in range(N)]
X[1]=[random.randint(10,150) for i in range(N)]
print(X)

print('Задание c')
print(X)
IMT = X[1]/((X[0]/100)**2)

for j in range(N):
    if IMT[j]<18.5 or IMT[j]>40:
        X[0][j]=0
        X[1][j]=0

X2=X[:, ~np.all(X == 0, axis=0)]
X=X2.copy()
#print(X)

div = X[1]/X[0]
N=div.shape[0]
print(div)

import math
k = math.ceil(1 +3.322*math.log10(N))
print('по формуле Стерджесса число групп = ', k)
axesX=[]
h= (np.max(div) - np.min(div))/k
print('интервал для групп= ',h)
l = np.min(div)
for io in range(k):
    axesX.append(l)
    counter = 0
    for i in range(N):
        if (div[i]>=l and div[i]<l+h):
            counter+=1

    print ('{:.2f}'.format(l)+' - ' +'{:.2f}'.format(l+h)+ ' = ' + str(counter))
    l += h

print(axesX) #размер = числу групп
TwoDimArray = np.zeros([k,k],dtype=np.int)
Q=[]
T=[]
CountIll=[]
for c in range(k):
    q =random.randint(0,25)
    Q.append(q)
    #print('q для рандома = ', q)
    #print('генерим третью строчку')
    X[2] = div + random.randint(0,q*q)
    t = random.choice(X[2])
    T.append(t)
    #print('Значение для глюкозы = ', t )
    #print('генерим 4 строчку')
    X[3]=[(0,1)[X[2][i]>t] for i in range(N)]
    l = np.min(X[2]) #минимальное значение сгенерированной глюкозы
    w=0
    okrj=np.max(X[2])
    for w in range(k):
        for i in range(N):
            if X[2][i]>=l and X[2][i]<=l+h:
                if X[3][i]==1:
                    TwoDimArray[c][w]+=1
        l+=h
    CountIll.append(np.ndarray.tolist(X[3]).count(1))
sumArray = np.sum(TwoDimArray,axis = 1)
for i in range(k):
    if CountIll[i]!=sumArray[i]:
        TwoDimArray[i][k-1]+=CountIll[i]-sumArray[i];
print(TwoDimArray)
from matplotlib import cm
fig = plt.figure()
ax = plt.axes(projection="3d")
Cucu1,Cucu2 = np.meshgrid(axesX,T)
ax.plot_wireframe(Cucu1, Cucu2, TwoDimArray,cmap='viridis')
ax.minorticks_on()
ax.set_xlabel('Разделение на группы')
ax.set_ylabel('Граница для глюкозы')
ax.set_zlabel('Количество заболевших')


fig2,ax2=plt.subplots()
ax2=plt.axes(projection='3d')
ax2.plot_surface(Cucu1, Cucu2, TwoDimArray, cmap=cm.coolwarm)
ax2.set_xlabel('Разделение на группы')
ax2.set_ylabel('Граница для глюкозы')
ax2.set_zlabel('Количество заболевших')
import matplotlib.ticker as ticker
fig2,ax3=plt.subplots()
ax3=plt.axes(projection='3d')
ax3.scatter(Cucu1, Cucu2, TwoDimArray, cmap='viridis')
ax3.set_xlabel('Разделение на группы')
ax3.set_ylabel('Граница для глюкозы')
ax3.set_zlabel('Количество заболевших')
plt.show()
# print('генерим третью строчку')
# q=random.randint(0,25)
# print('q для рандома = ', q)
# newArray = div + random.randint(0,q*q)
#
# k = math.floor(1 +1.44*math.log1p(N))
# print('по формуле Стерджесса число групп = ', k)
#
# h= (np.max(newArray) - np.min(newArray))/k
# print('интервал для групп= ',h)
# l = np.min(newArray)
# while l<np.max(newArray):
#     counter = 0
#     for i in range(N):
#         if (newArray[i]>=l and newArray[i]<l+h):
#             counter+=1
#
#     print ('{:.2f}'.format(l)+' - ' +'{:.2f}'.format(l+h)+ ' = ' + str(counter))
#     l += h
#
# list1 = np.ndarray.tolist(newArray)
# t = random.choice(list1)
# print('Значение для глюкозы = ', t )
# list2=[(0,1)[list1[i]>t] for i in range(N)]
# print('количество заболевших = ', list2.count(1))
# print('again')
# q=random.randint(0,25)
# print('q для рандома = ', q)
# newArray = div + random.randint(0,q*q)
#
# k = math.floor(1 +1.44*math.log1p(N))
# print('по формуле Стерджесса число групп = ', k)
#
# h= (np.max(newArray) - np.min(newArray))/k
# print('интервал для групп= ',h)
# l = np.min(newArray)
# while l<np.max(newArray):
#     counter = 0
#     for i in range(N):
#         if (newArray[i]>=l and newArray[i]<l+h):
#             counter+=1
#
#     print ('{:.2f}'.format(l)+' - ' +'{:.2f}'.format(l+h)+ ' = ' + str(counter))
#     l += h
# list1 = np.ndarray.tolist(newArray)
# t = random.choice(list1)
# print('Значение для глюкозы = ', t )
# list2=[(0,1)[list1[i]>t] for i in range(N)]
# print('количество заболевших = ', list2.count(1))
