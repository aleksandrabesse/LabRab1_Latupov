import cv2
img = cv2.imread('sq.jpg')
print('Количество пикселей = ', img.shape[0],'*',img.shape[1])
print('Задание 1')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('Изображение в серых тонах')
print(gray)
print('Задание 2')
string = gray[int(256/2)]
print('Центральная строка')
print(string)
print('Задание 3')
temp =list(map(round,string/20));
X = [i*20 for i in temp]
print('Строка после квантования')
print(X)

import numpy as np;
hist = np.histogram(X,bins=np.arange(min(X), max(X)+2))
print('Задание 4')
freq = hist[0]
value=list(map(int,hist[1]))
freqWithountNull=[]
for i in range(len(freq)):
    if freq[i]!=0:
        freqWithountNull.append(int(value[i]))
        print('Частота появления символа '+str(value[i])+' = '+str(freq[i]/256) + ' ('+str(freq[i])+' из 256)')

haff=[]
for i in range(len(freq)):
    if freq[i]!=0:
       haff.append({'sym': value[i], 'freq': freq[i], 'code': -1})
haff=sorted(haff,key=lambda i:i['freq'])

freqSentence=[]
freqNew=[]
left=True;
haff2=sorted(haff,key=lambda i:i['sym'])
print('Задание 5')
while (len(haff)>0 or len(freqNew)>0):
    min1={'sym': '', 'freq': np.inf , 'code': -1}
    min2={'sym': '', 'freq': np.inf , 'code': -1}
    min3 = {'sym': '', 'freq': np.inf , 'code': -1}
    min4={'sym': '', 'freq': np.inf , 'code': -1}
    if len(haff)>=2:
            min1=haff[0]
            haff.remove(min1)
            min2=haff[0]
            haff.remove(min2)
    elif len(haff)==1:
            min1=haff[0]
            haff.clear()
    if len(freqNew)>=2:
            min3 = freqNew[0]
            freqNew.remove(min3)
            min4 = freqNew[0]
            freqNew.remove(min4)
    elif len(freqNew)==1:
            min3=freqNew[0]
            freqNew.clear()
    if min1==min([min1,min2,min3,min4],key=lambda i:i['freq']):
        if min2==min([min2,min3,min4],key=lambda i:i['freq']):
            if min3['freq']!=np.inf:
                freqNew.insert(0,min3)
                if min4['freq']!=np.inf:
                    freqNew.insert(1,min4)
            minF = min1;
            minS=min2;
        elif min3==min([min2,min3,min4],key=lambda i:i['freq']):
            if min2['freq']!=np.inf:
                haff.insert(0,min2)
            if min4['freq']!=np.inf:
                freqNew.insert(0, min4)
            minF=min1;
            minS=min3;
        else:
            if min2 ['freq']!= np.inf:
                haff.insert(0, min2)
            if min3 ['freq']!= np.inf:
                freqNew.insert(0, min3)
            minF=min1;
            minS=min4;
    elif min2==min([min1,min2,min3,min4],key=lambda i:i['freq']):
        minF=min2;
        if min1==min([min1,min3,min4],key=lambda i:i['freq']):
            if min3['freq']!=np.inf:
                freqNew.insert(0,min3)
                if min4['freq']!=np.inf:
                    freqNew.insert(1,min4)
            minS=min1
        elif min3==min([min1,min3,min4],key=lambda i:i['freq']):
            if min1['freq']!=np.inf:
                haff.insert(0,min1)
            if min4['freq']!=np.inf:
                freqNew.insert(0, min4)
            minS=min3;
        else:
            if min1['freq'] != np.inf:
                haff.insert(0, min1)
            if min3 ['freq']!= np.inf:
                freqNew.insert(0, min3)
            minS=min4;
    elif min3==min([min1,min2,min3,min4],key=lambda i:i['freq']):
        minF=min3;
        if min1==min([min1,min2,min4],key=lambda i:i['freq']):
            minS=min1;
            if min2['freq']!=np.inf:
                haff.insert(0,min2)
            if min4['freq']!=np.inf:
                freqNew.insert(0,min4)
        elif min2==min([min1,min2,min4],key=lambda i:i['freq']):
            minS=min2;
            if min1['freq']!=np.inf:
                haff.insert(0,min1)
            if min4['freq']!=np.inf:
                freqNew.insert(0, min4)
        else:
            minS=min4
            if min1['freq']!= np.inf:
                haff.insert(0, min1)
            if min2['freq']!= np.inf:
                haff.insert(1, min2)
    else:
        minF=min4;
        if min1==min([min1,min2,min3],key=lambda i:i['freq']):
            minS=min1;
            if min2['freq']!=np.inf:
                haff.insert(0,min2)
            if min3['freq']!=np.inf:
                freqNew.insert(0,min3)
        elif min2==min([min1,min2,min3],key=lambda i:i['freq']):
            minS=min2;
            if min1['freq']!=np.inf:
                haff.insert(0,min1)
            if min3['freq']!=np.inf:
                freqNew.insert(0, min3)
        else: #min3
            minS=min3;
            if min1['freq']!= np.inf:
                haff.insert(0, min1)
            if min2['freq']!= np.inf:
                haff.insert(1, min2)

    if (minF['freq']+minS['freq']>256):
        break;
    newNote = {'sym': '', 'freq': minF['freq'] + minS['freq'], 'code': -1}
    if minS['sym']!='' and minF['sym']!='':
        if left:
            minF['code']='l'
            minS['code']='r'
            newNote['code']='l'
            left=False;
        else:
            minF['code'] = 'l'
            minS['code'] = 'r'
            newNote['code']='r'
    elif minF['sym']=='' and minF['code']=='l':
        minS['code']='r'
        newNote['code']='l'
    elif minS['sym']=='' and minS['code']=='r':
        minF['code'] = 'l'
        newNote['code'] = 'r'
    elif minS['sym']=='' and minS['code']=='l':
        minF['code']='r'
        newNote['code']='l'
        minS,minF=minF,minS
    elif minF['sym']=='' and minF['code']=='r':
        minS['code'] = 'l'
        newNote['code'] = 'r'
        minS, minF = minF, minS
    freqSentence.append(minF)
    freqSentence.append(minS)

    freqSentence.append(newNote)
    freqNew.append(newNote)
    #freqNew=sorted(freqNew, key = lambda i:i['freq']).copy()
    # haff = sorted(haff, key=lambda i: i['freq']).copy()

for i in freqSentence:
    if i['sym']!='':
        code=''
        copyI=i;
        k=-1
        while (k<len(freqSentence)-1):
            k = freqSentence.index(copyI,k+1)
            if k%3==0:
                code+=str(0)
                k+=2
            else:
                code+=str(1)
                k+=1
            if k<len(freqSentence):
                copyI=freqSentence[k]
        for k in haff2:
            if k['sym']==i['sym']:
                k['code']=code[::-1]
                k['freq']/=256
                print('Код символа '+str(k['sym'])+' - '+str(k['code']))
print('Задание 6')
print('Закодированная центральная строка после квантования')
for i in X:
    print(haff2[freqWithountNull.index(i)]['code'],end='')
print()
print('Задание 7')
L=0
for i in haff2:
    L+=len(i['code'])*i['freq']
print('Средняя ожидаемая длина = ',L)
import math
Hx=0
for i in haff2:
    Hx+=i['freq']*math.log2(1/i['freq'])
print('Энтропия источника = ',Hx)
print('Итоги:')
if L>=Hx and L<=Hx+1:
    print('Кодирование оптимально, так как L принадлежит [H,H+1]')
else:
    print('Кодирование неоптимальное')