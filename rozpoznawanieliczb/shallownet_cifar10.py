from pandas import test
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

print('[INFO] ładowanie danych z CIFAR10...\n')
((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX=trainX.astype('float')/255.0
testX=testX.astype('float')/255.0

lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.fit_transform(testY)
labelNames=['samolot','samochód','ptak','kot','jeleń','pies',
            'żaba','koń','statek','ciężarówka']

print('[INFO] przygotowanie modelu...')
opt= SGD(lr=0.01)
model=ShallowNet.build(width=32, height=32, depth=3,
                       classes=10)
model.compile(loss='categorical_crossentropy',optimizer=opt,
              metrics=['accuracy'])

print('[INFO]- trenowanie sieci...')
H=model.fit(trainX,trainY,validation_data=(testX,testY),
            batch_size=32, epochs=40,verbose=1)

print('[INFO]- ewaluacja sieci...')
predictions=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1), 
                            predictions.argmax(axis=1),target_names=labelNames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,40), H.history['loss'], label='strata trenowania')
plt.plot(np.arange(0,40), H.history['val_loss'], label='strata wart. trenowania')
plt.plot(np.arange(0,40), H.history['accuracy'], label='dopasowanie')
plt.plot(np.arange(0,40), H.history['val_accuracy'], label='wart. dopasowania')
plt.title('Dane historyczne trenowania modelu')
plt.xlabel('Epoka #')
plt.ylabel('strata/dopasowanie')
plt.legend()
plt.show()