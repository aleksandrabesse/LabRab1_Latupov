# ctrl alt l
import pandas
from numba import njit, prange

df = pandas.read_csv('titanic_data.csv')  # читаем данные из файла
import numpy as np

data = df[:len(df) - 1]  # удаляем последнюю строку, она нулевая
count = data.shape[0]  # размер входных данных
yData = np.array(data['Survived'].tolist(), dtype=np.int)  # массив меток, yi
cols = [col for col in list(data) if col not in ['Name']]
temp = [data[cols[k]].tolist() for k in range(1, len(cols))]
xData = np.array([temp], dtype=np.float).T  # массив шестиразрядных векторов, xi


# @njit(fastmath=True)
# def dot(a, b):
#     mulp = 0;
#     for i in range(len(a)):
#         mulp += a[i] * b[i];
#     return mulp;
#
#
# o = np.ones((1, 5))
# o1 = 2 * np.ones((5, 1))
# print(dot(o[0], o1))


@njit(fastmath=True)
def countFunctionGradient(x, y, z):
    z = z.T
    if y == 0:
        mulpWithPlus = np.dot(z, x)[0]

        return x * mulpWithPlus / (1 + mulpWithPlus)
    else:
        mulpWithMinus = np.dot(-z, x)[0]

        return x * np.exp(mulpWithMinus) / (1 + mulpWithMinus)


def gradientFunction(count, b):
    sum = 0
    for i in range(count):
        sum += countFunctionGradient(xData[i], yData[i], b)
    return sum


    # @ njit(fastmath=True)


def countLogicalRegressionFunction(x, y, z):
    z = z.T
    if y==1:
        mulpWithMinus = np.dot(-z, x)[0]
        print('y==1')
        return np.log2(1/(1+np.exp(mulpWithMinus)))
    else:
        mulpWithPlus = np.dot(z, x)[0]
        print('y==0')
        return np.log2(1/(1+np.exp(mulpWithPlus)))


# @njit(fastmath=True)
def logicalRegressionFunction(count, b):
    sum = 0
    for i in range(count):
        sum += countLogicalRegressionFunction(xData[i], yData[i], b)
    return sum


from datetime import datetime

print(datetime.now())
from sympy import *

# print(data.isnull().sum())  # проверка на пропущенные данныек
x, y, z = symbols('x y z')
function = log(((1 / (1 + exp(-z * x))) ** (y)) * ((1 / (1 + exp(z * x))) ** (1 - y)))
gradient = diff(function, z)
print(gradient)
answers = []
countIterations = 1000000

import random

curr = random.choice(xData)
k = gradientFunction(int(0.8 * count), curr)
print(k)
b = curr+0.01*k
print(b)
log = logicalRegressionFunction(int(0.8 * count), b)
print(log)
def mainFunction():
    global answers
    currentVector = np.zeros((6, 1))
    step = 0.000001
    for i in range(countIterations):
        k = gradientFunction(int(0.8 * count), currentVector)
        b = currentVector + step * k
        answers.append((currentVector, logicalRegressionFunction(int(0.8 * count), b)))
        currentVector = b


# mainFunction()

# maximum = answers[0]
# ans = []
# for i in answers:
#     ans.append(i[1])
#     if i[1] > maximum[1]:
#         maximum = i
# print(maximum)
#
# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
#
# # syntax for 3-D projection
# ax = plt.axes()
# print(datetime.now())
# ax.scatter(range(0, countIterations), ans)
# plt.show()
