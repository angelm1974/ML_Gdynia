from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_poch(x):
    return x * (1 - x)


def predykcja(X,W):
    pred=sigmoid_activation(X.dot(W))
    
    pred[pred <= 0.5]=0
    pred[pred > 0]=1
    
    return pred

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epoki', type=float, default=100,help='# epok')
ap.add_argument('-a', '--alfa', type=float, default=0.01,help='# alfa, dokładność')
args=vars(ap.parse_args())

(X,y)=make_blobs(n_samples=10000,n_features=2,centers=2, cluster_std=1.5, random_state=1)
y=y.reshape((y.shape[0],1))

X=np.c_[X, np.ones((X.shape[0]))]
(trainX,testX,trainY,testY)=train_test_split(X,y,test_size=0.5, random_state=42)

print('[INFO] trenowanie...')
W =np.random.rand(X.shape[1],1)
straty=[]

for epoka in np.arange(0,args['epoki']):
    pred =sigmoid_activation(trainX.dot(W))
    
    error= pred -trainY
    strata=np.sum(error ** 2)
    straty.append(strata)
    
    d = error * sigmoid_poch(pred)
    gradient=trainX.T.dot(d)
    
    W+= -args['alfa']*gradient
    
    if epoka ==0 or(epoka + 1) %5 ==0:
        print('[INFO] epoka={}, strata={:.7f}'.format(int(epoka+1),strata))
        
print('[INFO] ewaluacja...')
pred=predykcja(testX,W)
print(classification_report(testY, pred))

plt.style.use('ggplot')
plt.figure()
plt.title("Dane")
plt.scatter(testX[:,0], testX[:,1], marker='o',c=testY[:,0],s=30)    

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,args['epoki']),straty)
plt.title('Funkcja straty podczas trenowania')
plt.xlabel('Epoka #')
plt.ylabel('Strata')
plt.show()    
    
