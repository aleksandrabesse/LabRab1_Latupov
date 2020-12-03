# ctrl alt l
import random
import matplotlib.pyplot as plt
import pandas
from numba import njit
from datetime import datetime
import numpy as np

df = pandas.read_csv('titanic_data.csv')  # читаем данные из файла

data = df[:len(df) - 1]  # удаляем последнюю строку, она нулевая
count = data.shape[0]  # размер входных данных
yData = np.array(data['Survived'].tolist(), dtype=np.int)  # массив меток, yi
cols = [col for col in list(data) if col not in ['Name']]
temp = [data[cols[k]].tolist() for k in range(1, len(cols))]
xData = np.array([temp]).T  # массив шестиразрядных векторов, xi


@njit(fastmath=True)
def dot(a, b):
    mulp = 0
    for i in range(a.shape[1]):
        mulp += a[0][i] * b[i][0]
    return mulp


@njit(fastmath=True)
def sigmoid(a):
    return 1.0 / (1 + np.exp(-a))


@njit(fastmath=True)
def countFunctionGradient(x, y, z):
    z = z.T
    temp = sigmoid(dot(z, x))
    if y == 0:
        return -temp * x
    else:
        return (1 - temp) * x


def gradientFunction(count, b):
    sum = 0
    for i in range(count):
        sum += countFunctionGradient(xData[i], yData[i], b)
    return sum


@njit(fastmath=True)
def countLogicalRegressionFunction(x, y, z):
    z = z.T
    commonElement = sigmoid(dot(z, x))
    if y == 0:
        return np.log2(1 - commonElement)
    else:
        return np.log2(commonElement)


def logicalRegressionFunction(count, b):
    sum = 0
    for i in range(count):
        sum += countLogicalRegressionFunction(xData[i], yData[i], b)
    return sum

def classify(x):
    temp = sigmoid(dot(b.T, x))
    if temp > 1 - temp:
        return 1
    else:
        return 0

# colvo = []
# stepik = []
# maximus = []

def mainFunction(step, curr):
    # global colvo, stepik, maximus
    answers = []
    countIterations = 100000
    currentTime = datetime.now()
    for i in range(countIterations):
        b = curr + step * gradientFunction(int(0.8 * count), curr)
        o = logicalRegressionFunction(int(0.8 * count), b)
        if o == -np.inf:
            i -= 1
        else:
            answers.append((curr, o))
        curr = b
    # if len(answers) > 0:
    maximum = answers[0]
    ans = []
    for i in answers:
        ans.append(i[1])
        if i[1] > maximum[1]:
            maximum = i
    print('Шаг', step, 'Количество значений', len(answers), 'Время выполнения',
            str((datetime.now() - currentTime).total_seconds()), 'Максимальное значение функции ', maximum[1],
            ' с вектором тэта  ', maximum[0])
        # fig = plt.figure()
        # ax = plt.axes()
        # ax.scatter(range(0, len(ans)), ans)
        # ax.set(xlabel='Итерации', ylabel='Значения функции l(q)',
        #        title='Шаг=' + str(step))
        # fig.savefig("test" + str(step) + ".png")
        # stepik.append(step)
        # colvo.append(len(answers))
        # maximus.append(maximum[1])


# Подбор шага. Этап 1
# steps = [0.0000001, 0.0000003, 0.0000005,0.0000006,0.0000007, 0.0000008, 0.0000009,0.000001, 0.000003, 0.000005, 0.000007, 0.000009,0.00001, 0.00003, 0.00005, 0.00007, 0.00009,0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.008, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1,
#          0.3, 0.5,
#          0.7, 0.9, 1]

# steps = [0.0000001, 0.0000003, 0.0000005, 0.0000006, 0.0000007, 0.0000008, 0.0000009, 0.000001, 0.000003, 0.000005]
# curr = random.choice(xData)
# for step in steps:
step = 5 * 10**(-6)
mainFunction(step, np.zeros((6, 1)))

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(colvo, stepik)
# ax.set(xlabel='Количество значений', ylabel='Шаг')
# fig.savefig("test.png")
# fig2 = plt.figure()
# ax2 = plt.axes()
# ax2.plot(stepik, maximus)
# ax2.set(xlabel='Шаг', ylabel='Максимум')
# fig2.savefig("max.png")


