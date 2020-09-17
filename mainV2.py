import random
import numpy as np
import matplotlib.pyplot as plt


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

N=X.shape[1]


print('Задание d')
fig,ax1=plt.subplots()

ax1.set_title('График роста и веса')
ax1.set_xlabel('Рост')
ax1.set_ylabel('Вес')
plt.grid(True)
ax1.scatter(X[0],X[1],marker='o')
#plt.show()

fig.savefig('Задание с.png',bbox_inches='tight',dpi=700)

print('Задание e')
div = X[1]/X[0]
fig,ax2=plt.subplots()
ax2.hist(div)
ax2.set_title('Гистограмма соотношения вес/рост')
plt.grid(True)
fig.savefig('Задание e.png',bbox_inches='tight',dpi=700)




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
u=5

ROW3=[]
ROW4=[]


for c in range(k):
    q =random.randint(0,u*u)
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
    ROW3.append(X[2].copy())
    ROW4.append(X[3].astype('int32').copy())


sumArray = np.sum(TwoDimArray,axis = 1)
for i in range(k):
    if CountIll[i]!=sumArray[i]:
        TwoDimArray[i][k-1]+=CountIll[i]-sumArray[i];


fig, ax6 = plt.subplots(2,5)
for i in range(5):
    color=[('red','green')[ROW4[i][o]] for o in range(N)]
    ax6[0][i].bar(np.arange(0,N),ROW3[i],color=color)
    ax6[0][i].set_xlabel('t = ' + '{:.2f}'.format(T[i]) + ' q = '+ '{:.2f}'.format(Q[i]))
    ax6[0][i].xaxis.set_label_position("top")
    ax6[0][0].set_ylabel('Уровень глюкозы')
for i in range(5):
    ax6[1][i].pie([CountIll[i],N-CountIll[i]],labels=['Больные','Здоровые'], autopct='%1.2f%%',colors=['r','g'])
fig.savefig('Задание h(1).png',dpi=1000)

from matplotlib import cm
Cucu1,Cucu2 = np.meshgrid(axesX,T)
fig2,ax2=plt.subplots()
ax2=plt.axes(projection='3d')
ax2.plot_surface(Cucu1, Cucu2, TwoDimArray, cmap=cm.coolwarm)
ax2.set_xlabel('Группы')
ax2.set_ylabel('Граница для глюкозы')
ax2.set_zlabel('Количество заболевших')
fig2.savefig('Задание h(2).png',dpi=700)

fig = plt.figure()
ax7 = fig.add_subplot(projection='3d')
for c in range(k):
    ax7.bar(axesX, TwoDimArray[c], zs=Q[c], zdir='y', alpha=1, width=0.05, label = 't = '+ '{:.2f}'.format(T[c]))
ax7.legend()
plt.xticks([round(axesX[i],2) for i in range(k)])

ax7.set_xlabel('Разделение по группам')
ax7.set_ylabel('Шум,q')
ax7.set_zlabel('Количество заболевших')
fig.savefig('Задание h(3).png',dpi=700)
plt.show()
