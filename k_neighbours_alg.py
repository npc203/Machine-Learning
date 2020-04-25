import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)
    confidence=vote_result[0][1]/k
    return vote_result[0][0],confidence

if __name__=="__main__":
    style.use('fivethirtyeight')
    dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
    new_features = [5,7]

    result = k_nearest_neighbors(dataset, new_features)
    print(result)
    plt.scatter(new_features[0], new_features[1], s=100, color = result)  
    [[plt.scatter(ii[0],ii[1],color=i) for ii in dataset[i]] for i in dataset]
    plt.show()
