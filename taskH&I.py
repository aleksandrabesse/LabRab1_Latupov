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
#Insertion()
#print(X)


# print('Задание d')
# fig,ax1=plt.subplots()
#
# ax1.set_title('График роста и веса')
# ax1.set_xlabel('Рост')
# ax1.set_ylabel('Вес')
# plt.grid(True)
# ax1.scatter(X[0],X[1],marker='o')
# #plt.show()
#fig.savefig('Задание с.png',bbox_inches='tight',dpi=700)

print('Задание e')
div = X[1]/X[0]
# fig,ax2=plt.subplots()
# ax2.hist(div)
# ax2.set_title('Гистограмма соотношения вес/рост')
# plt.grid(True)
# #plt.show()
# fig.savefig('Задание e.png',bbox_inches='tight',dpi=700)

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
#row2,row3=Insertion2(row2,row3,ColVo)

# fig,ax3=plt.subplots()
# legend=[]
# for i in range(ColVo):
#     ax3.plot(row2[i],row3[i])
#     legend.append('{:.2f}'.format(t[i]))
# ax3.legend(legend)
# ax3.set_title('Сгруппированные данные для различных значений шума и порогового значения(в. 1)')
# plt.grid(True)
# fig.savefig('Задание h1.png',bbox_inches='tight',dpi=700)

print('Задание i')

ColVo=5
trash = np.arange(0,ColVo)
R=[]
for i in trash:
    R.append(div + i ** 2)
t=[random.choice(k) for k in R]
R2=[[0 for i in range(N)] for j in range(ColVo)]
for i in range(ColVo):
    for j in range(N):
        R2[i][j]=R[i][j]<t[i]

from mpl_toolkits.mplot3d import Axes3D

# mass = [5 for i in range(100)]
# R = div[:100] + random.randint(0,25)
# u = random.choice(R)
# mass2=[u for i in range(100)]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection ='3d')
# ax.scatter(mass,mass2,R)
# mass = [6 for i in range(100)]
# R = div[:100] + random.randint(0,36)
# u = random.choice(R)
# mass2=[u for i in range(100)]
# ax.scatter(mass,mass2,R)

# plt.show()
# print(mass)
# print(mass2)
# print(R)
#Axes3D

Mass1 = np.arange(0,10)
Third=[div[:10] + random.randint(0,5) for i in range(10)]
Mass2=[random.choice(Third[i]) for i in range(10)]
Fort=[Third[i]>Mass2[i] for i in range(10)]
print(Mass1)
print(Third)
print(Mass2)
print(Fort)

Cucu1,Cucu2 = np.meshgrid(Mass1,Mass2)
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('Значение для шума')
ax.set_ylabel('Значение для порога')
ax.set_zlabel('Уровень глюкозы')
Array = np.array(Third)
ax.plot_surface(Cucu1, Cucu2,Array , color='green')
plt.show()

Cucu1,Cucu2 = np.meshgrid(Mass1,Mass2)
# fig = plt.figure()
# ax = plt.axes(projection="3d")
#
# ax.set_xlabel('Значение для шума')
# ax.set_ylabel('Значение для порога')
# ax.set_zlabel('Уровень глюкозы')
# Array = np.array(Third)
# ax.bar3d()
# ax.bar3d(Cucu1, Cucu2,Array , color='green')
# plt.show()