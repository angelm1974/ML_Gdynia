from algorytmy.nn import Perceptron
import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[0],[0],[1]])

print('[INFO] trenowanie perceptrona...')
p=Perceptron(X.shape[1],alpha=0.1)
p.fit(X,y,epoki=20)

print('[INFO] testowanie perceptrona...')

for (x,target) in zip(X,y):
    pred=p.predict(x)
    print('[INFO] dane={}, prawda_podstawowa={}, predykcja={}'.format(x,target[0],pred))
    