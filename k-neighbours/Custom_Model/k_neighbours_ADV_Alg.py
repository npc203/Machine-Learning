import numpy as np
import pandas as pd
import random
import sys
import warnings
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import style
import matplotlib.pyplot as plt

class k_nearest_neighbors():    
    def __init__(self):
        #print("init")
        self.weights=[]
        self.k=0
        self.accuracy=None

    def fit(self,data,label,k=3):
        self.k=k
        #weights=[]
        if len(Counter(label)) >= k:
            warnings.warn('K is set to a value less than total voting groups!')   
        distances = []
        for i in range(len(data)):
            for features in data:        
                euclidean_distance = np.linalg.norm(np.array(features)-np.array(data[i]))
                distances.append([euclidean_distance,label[data.index(features)]])
            votes = [i[1] for i in sorted(distances)[:k]]
            self.weights.append([data[i],Counter(votes).most_common(1)[0][0]])
            distances=[]
        #print(weights)
        
    def score(self,data,labels):
        check=self.predict(data)
        tot,rit=0,0
        for i in range(len(labels)):
            if labels[i]==check[i][0]:
                rit+=1
            tot+=1
        return (rit/tot)*100

    def predict(self,test):
        k=self.k
        weights=self.weights
        result=[]
        distances=[]
        for i in test:
            for features in weights:        
                    euclidean_distance = np.linalg.norm(np.array(features[0])-np.array(i))
                    distances.append([euclidean_distance,features[1]])
                    votes = [minimum[1] for minimum in sorted(distances)[:k]]
            result.append([Counter(votes).most_common(1)[0][0],Counter(votes).most_common(1)[0][1]/k])
            #print(distances,'##')
            distances=[]
        return result
    
if __name__=="__main__":
    style.use('fivethirtyeight')
    data =[[1,2,'k'],[2,3,'k'],[3,1,'k'],[6,5,'r'],[7,7,'r'],[8,6,'r']]
    random.shuffle(data)
    X=[i[:-1]for i in data]
    y=[i[-1]for i in data]
    new_features = [[5,7],[0,4],[10,10]]
    new_labels=['r','k','r']
    clf=k_nearest_neighbors()
    clf.fit(X,y,k=3)
    print("Accuracy:",clf.score(new_features,new_labels))
    result=clf.predict(new_features)
    print(new_features)
    #result contains the prediction and the confidence of the prediction 
    print(result)
    for i in new_features:
        plt.scatter(i[0], i[1], s=100, color = result[new_features.index(i)][0])  
    for i in data:
        plt.scatter(i[0],i[1],color=i[2])
    plt.show()

