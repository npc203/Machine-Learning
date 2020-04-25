import cv2
import os 
import numpy as np

PATH='draw/'
emt=[]

for i in os.listdir(PATH):
   im=cv2.imread(PATH+i)
   im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
   im=cv2.resize(im,(28,28))
   emt.append(im)

emt=np.asarray(emt)
print(emt.shape)
