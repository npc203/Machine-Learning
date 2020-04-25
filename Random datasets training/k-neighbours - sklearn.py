import numpy as np
from sklearn import preprocessing,neighbors,svm
from sklearn.model_selection import train_test_split
import pandas as pd

#buying,maint,doors,persons,lug_boot,safety,class

df = pd.read_csv(r'car\car.data')

df.replace("vhigh",3,inplace=True)
df.replace("high",2,inplace=True)
df.replace("big",2,inplace=True)
df.replace("med",1,inplace=True)
df.replace("low",0,inplace=True)
df.replace("more",-99999,inplace=True)
df.replace("5more",-99999,inplace=True)
df.replace("small",0,inplace=True)

df.replace("unacc",0,inplace=True)
df.replace("acc",1,inplace=True)
df.replace("good",2,inplace=True)
df.replace("vgood",3,inplace=True)


df = df.astype(float)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#clf = neighbors.KNeighborsClassifier()
clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('accuracy:'+str(accuracy*100))

example=np.array([[0,2,4,8,1,2],[0,0,0,0,0,0]])
example=example.reshape(len(example),-1)
predic=clf.predict(example)
print(predic)
