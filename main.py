import random
import numpy as np
import matplotlib.pyplot as plt


D=4
N = random.randint(1000,2000)
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
trash = [random.randint(0,10) for i in range(5)]#float(input('Введите o'))
print('Задание g')
t = []#float(input('Введите t'))


row2=[]
row3=[]
for step in range(5):
    def generate():
        global X
        X[2] = div + random.randint(0, trash[step] ** 2)
        t.append(np.median(X[2]))
        for i in range(N):
            if X[2][i] < t[step]:
                X[3][i] = 0
            else:
                X[3][i] = 1
    generate()
    row2.append(np.copy(X[2]))
    row3.append(np.copy(X[3]))

def Insertion2():
    global row2,row3
    for y in range(5):
        for i in range(1,N):
            j=i
            while j>0 and row2[y][j-1]>row2[y][j]:
                row2[y][j-1],row2[y][j]=row2[y][j],row2[y][j-1]
                row3[y][j-1],row3[y][j]=row3[y][j],row3[y][j-1]
                j-=1;
Insertion2()

fig,ax3=plt.subplots()
ax3.plot(row2[0],row3[0],row2[1],row3[1],row2[2],row3[2],row2[3],row3[3],row2[4],row3[4])
ax3.set_title('Сгруппированные данные для различных значений шума и порогового значения')
plt.grid(True)
plt.show()



print('Задание h')
print('Задание i')