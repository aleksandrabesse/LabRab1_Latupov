import io
book=[]
with io.open('Zvezdnaya_Gorod-drakonov_1_Gorod-drakonov.B9FTmg.560032.txt', encoding='utf-8') as file:
    book= [line.strip() for line in file.readlines()]
i=0;
bookOneString = ''
while i<len(book):
    if book[i]=='':
        del book[i];
    else:
        bookOneString += book[i] + ' '
        i+=1;

print(bookOneString)

sym = []
N = len(bookOneString);
for i in range(N):
    if bookOneString[i] not in sym:
        sym.append(bookOneString[i])
#print(sym)
ColVoXi=[]
for i in sym:
    ColVoXi.append(bookOneString.count(i))
print(ColVoXi)

probability=[]
for i in ColVoXi:
    probability.append(i/N);
print(probability)

import pandas as pd
data=pd.DataFrame(probability)
print(data.describe())

import math
Hx=0
for i in probability:
    Hx+=i*math.log2(1/i)
print('Энтропия символа = ', Hx)
CommonEntropy=Hx*N
print('Общая энтропия текста= ', CommonEntropy)

import cv2
img = cv2.imread('43678063-elena-zvezdnaya-gorod-drakonov.jpg')
N2 = img.shape[0]*img.shape[1];
print('Количество пикселей = ', N2)

#переводим в HSV
#hsv_img = rgb2hsv(img)
#saturation = hsv_img[...,1];
#print(saturation)
###

#в серые тона
import cv2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray)

ColVoXi2=[0 for i in range(256)]
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        ColVoXi2[gray[i][j]]+=1
print(ColVoXi2)

probability2=[]
for i in ColVoXi2:
    probability2.append(i/N2)
print(probability2)

data=pd.DataFrame(probability2)
print(data.describe())
Hx2=0
for i in probability2:
    Hx2+=i*math.log2(1/i)
print('Энтропия пикселя = ', Hx2)
CommonEntropy2=Hx2*N2
print('Общая энтропия изображения= ', CommonEntropy2)
