import numpy as np
import cv2

labels=['pies','kot','sowa']
np.random.seed(0)

W=np.random.randn(3,3072)
b=np.random.randn(3)

original=cv2.imread("kot.jpeg")
obraz= cv2.resize(original,(32,32)).flatten()

scores=W.dot(obraz) + b

for(label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

cv2.putText(original, "Etykieta: {}".format(labels[np.argmax(scores)]),
            (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),3)

cv2.imshow("obraz",original)
cv2.waitKey(0)