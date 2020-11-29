# ctrl alt l
import pandas

df = pandas.read_csv('titanic_data.csv')  # читаем данные из файла
import numpy as np

data = df[:len(df) - 1]  # удаляем последнюю строку, она нулевая
count = data.shape[0]  # размер входных данных
yData = np.array(data['Survived'].tolist(), dtype=np.int)  # массив меток, yi
cols = [col for col in list(data) if col not in ['Name']]
temp = [data[cols[k]].tolist() for k in range(1, len(cols))]
xData = np.array([temp], dtype=np.float).T  # массив шестиразрядных векторов, xi


def countFunctionGradient(x, y, z):
    z = z.T
    first = y * x * np.exp(np.dot(-z, x)) / (1 + np.exp(np.dot(-z, x)))
    second = (1 - y) * x * np.exp(np.dot(z, x)) / (1 + np.exp(np.dot(z, x)))
    return first - second;


def gradientFunction(count, b):
    sum = 0
    for i in range(count):
        sum += countFunctionGradient(xData[i], yData[i], b)
    return sum


def countLogicalRegressionFunction(x, y, z):
    z = z.T.copy()
    return np.log2(
        ((1 / (1 + np.exp(np.dot(-z, x)[0][0]))) ** (y)) *
        ((1 / (1 + np.exp(np.dot(z, x)[0][0]))) ** (1 - y)))


def logicalRegressionFunction(count, b):
    sum = 0
    for i in range(count):
        sum += countLogicalRegressionFunction(xData[i], yData[i], b)
    return sum
from datetime import datetime
print(datetime.now())
from sympy import *

print(data.isnull().sum())  # проверка на пропущенные данныек
x, y, z = symbols('x y z')
function = log(((1 / (1 + exp(-z * x))) ** (y)) * ((1 / (1 + exp(z * x))) ** (1 - y)))
gradient = diff(function, z)

currentVector = np.ones((6, 1))
currentVector*=-1
answers = []
countIterations = 1000
step = 0.00009
for i in range(countIterations):
    b = currentVector + step * gradientFunction(int(0.8 * count), currentVector)
    answers.append((currentVector, logicalRegressionFunction(int(0.8 * count), b)))
    currentVector = b
print(step)
maximum = answers[0]
ans=[]
for i in answers:
    ans.append(i[1])
    if i[1] > maximum[1]:
        maximum = i
print(maximum)

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes()
print(datetime.now())
ax.scatter(range(0,countIterations),ans)
plt.show()