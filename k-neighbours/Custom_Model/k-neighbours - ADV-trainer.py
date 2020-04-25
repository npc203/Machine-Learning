import numpy as np
import pandas as pd
import random
import time
import pickle
from sklearn.model_selection import train_test_split
from k_neighbours_ADV_Alg import k_nearest_neighbors


df = pd.read_csv(r'E:\sam\Machine Learning\k-neighbours\breast-cancer-wisconsin.data')
df.replace("?",-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

#print(full_data)

X=[i[:-1]for i in full_data]
y=[i[-1]for i in full_data]  

train_set,test_set,train_label,test_label=train_test_split(X,y,test_size=0.2)

#Train model
print("begin")
s=time.time()
clf=k_nearest_neighbors()
clf.fit(train_set,train_label,k=5)
e=time.time()
print("end...","time taken:",e-s)
pickle.dump(clf,open("k_neighbor_model.pickle","wb"))


