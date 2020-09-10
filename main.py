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
#IMT
IMT = X[1]/((X[0]/100)**2)

for j in range(N):
    if IMT[j]<18.5 or IMT[j]>40:
        X[0][j]=0
        X[1][j]=0

X2=X[:, ~np.all(X == 0, axis=0)]
X=X2.copy()
print(X)

def Insertion():
    global X
    for i in range(1,N):
        j=i
        while j>0 and X[0][j-1]>X[0][j]:
            X[0][j-1],X[0][j]=X[0][j],X[0][j-1]
            X[1][j-1],X[1][j]=X[1][j],X[1][j-1]
            j-=1;
N=X.shape[1]
Insertion()
print(X)


print('Задание d')
fig,ax1=plt.subplots()
ax1.set_title('График роста и веса')
ax1.set_xlabel('Рост')
ax1.set_ylabel('Вес')
plt.grid(True)
ax1.scatter(X[0],X[1],marker='o')
#plt.show()
#fig.savefig('Задание с.png')

print('Задание e')
div = X[1]/X[0]
fig,ax2=plt.subplots()
ax2.hist(div)
ax2.set_title('Гистограмма соотношения')
plt.grid(True)
#plt.show()
#fig.savefig('Задание e.png')

print('Задание f')
trash = [random.randint(0,10) for i in range(5)]
print('Задание g')
t = []

def generate(step):
    global X
    X[2] = div + random.randint(0, trash[step] ** 2)
    t.append(np.median(X[2]))
    for i in range(N):
        if X[2][i] < t[step]:
            X[3][i] = 0
        else:
            X[3][i] = 1
generate(0)

print('Задание h')

row2=[]
row3=[]
row2.append(np.copy(X[2]))
row3.append(np.copy(X[3]))
ColVo=5
for step in range(1,ColVo):
    generate(step)
    row2.append(np.copy(X[2]))
    row3.append(np.copy(X[3]))

def Insertion2(row2,row3,Col):
    for y in range(Col):
        for i in range(1,N):
            j=i
            while j>0 and row2[y][j-1]>row2[y][j]:
                row2[y][j-1],row2[y][j]=row2[y][j],row2[y][j-1]
                row3[y][j-1],row3[y][j]=row3[y][j],row3[y][j-1]
                j-=1;
    return row2,row3
row2,row3=Insertion2(row2,row3,ColVo)

fig,ax3=plt.subplots()
legend=[]
for i in range(ColVo):
    ax3.plot(row2[i],row3[i])
    legend.append('{:.2f}'.format(t[i]))
ax3.legend(legend)
ax3.set_title('Сгруппированные данные для различных значений шума и порогового значения')
plt.grid(True)


print('Задание i')

ColVo=5
trash = np.arange(0,ColVo)
R=[]
for i in trash:
    R.append(div + i ** 2)
#t=[np.median(k) for k in R]
t=[random.choice(k) for k in R]
R2=[[0 for i in range(N)] for j in range(ColVo)]
for i in range(ColVo):
    for j in range(N):
        R2[i][j]=R[i][j]<t[i]
#R2=[((1,0)[R[i][j]<t[i]] for j in range(N)) for i in range(ColVo)]

R,R2=Insertion2(R,R2,ColVo)
legend=[]
fig,ax4=plt.subplots()
for i in range(ColVo):
    ax4.plot(R[i],R2[i])
    legend.append('{:.2f}'.format(trash[i]))
ax4.legend(legend)
ax4.set_title('Данные для разного значения шума')
plt.grid(True)

fig,ax5=plt.subplots()
for k in range(ColVo):
    ax5.plot(np.arange(0,N),R[k])
    ax5.axhline(t[k],color='r',linewidth=1)


fig, ax6 = plt.subplots(1,ColVo)
for i in range(ColVo):
    ax6[i].bar(np.arange(0,N),R[i],color=[('red','green')[R2[i][o]] for o in range(N)])
    ax6[i].set_xlabel('t = ' + '{:.2f}'.format(t[i]))
    ax6[i].xaxis.set_label_position("top")
plt.show()
