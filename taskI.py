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
div = X[1]/X[0]

u=3
Q=np.arange(0,u*u)
X[2] = div + Q[0]
t = random.choice(X[2])
X[3] = [(0, 1)[X[2][i] > t] for i in range(N)]
colVoIll=np.ndarray.tolist(X[3]).count(1)

fig, ax = plt.subplots()
fig2,ax2=plt.subplots()
colVoIll=[]
for i in range(u*u):
    X[2] = div + Q[i]
    ax.plot(np.arange(0,N),X[2],label='q= '+ str(Q[i]))
    X[3] = [(0, 1)[X[2][i] > t] for i in range(N)]
    colVoIll.append(np.ndarray.tolist(X[3]).count(1))
ax.legend()
ax.set_title('Зависимость уровня глюкозы от значения шума')
ax.set_xlabel('Люди')
ax.set_ylabel('Уровень глюкозы')
fig.savefig('Задание i(1).png',dpi=700)
ax2.bar(Q,colVoIll)
ax2.set_title('Зависимость количества заболевших от значения шума')
ax2.set_xlabel('Значение шума')
ax2.set_ylabel('Количество заболевших')

fig2.savefig('Задание i(2).png',dpi=700)
colVoIll.clear()
u=4
T=[random.choice(X[2]) for i in range(u*u)]
T.sort()
for p in range(u*u):
    X[3] = [(0, 1)[X[2][i] > T[p]] for i in range(N)]
    colVoIll.append(np.ndarray.tolist(X[3]).count(1))
fig, ax3 = plt.subplots()
ax3.plot(T,colVoIll)
ax3.set_title('Зависимость количества заболевших от параметра t')
ax3.set_xlabel('Параметр t')
ax3.set_ylabel('Количество заболевших')
plt.show()