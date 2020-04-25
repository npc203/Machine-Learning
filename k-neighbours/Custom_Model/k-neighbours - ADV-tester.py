import numpy as np
import pandas as pd
import random
import time
import pickle
#from k_neighbours_ADV_Alg import k_nearest_neighbors


#Test model

clf=pickle.load(open("k_neighbor_model.pickle","rb"))
print("begin")
s=time.time()
#print(clf.score(test_set,test_label))

#2=not dangerous
#4=malignant

result=clf.predict([[1,7,1,6,7,2,1,2,7]])         
e=time.time()
print("end...","time taken:",e-s)
print(result)

