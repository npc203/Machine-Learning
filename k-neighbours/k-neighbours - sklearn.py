import numpy as np
from sklearn import preprocessing,neighbors,svm
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace("?",-999999,inplace=True)
df.drop(['id'],1,inplace=True)
df = df.astype(float)
#print(type(df['bare_nuclei'][1]))

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
#clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('accuracy:'+str(accuracy*100))

example=np.array([[4,2,1,1,1,2,3,2,1],[8,1,1,1,2,2,3,2,3]])
example=example.reshape(len(example),-1)
predic=clf.predict(example)
print(predic)
