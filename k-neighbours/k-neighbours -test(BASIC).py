import numpy as np
import pandas as pd
import random
import sys

sys.path.append(r"E:\sam\Machine Learning")
from k_neighbours_alg import k_nearest_neighbors

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace("?",-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

#print(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote,confidence=k_nearest_neighbors(train_set,data,k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:',(correct/total)*100)
print('Confidence:',confidence)
