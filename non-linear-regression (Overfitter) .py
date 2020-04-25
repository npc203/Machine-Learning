from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random
import codecs
import warnings
from unicodes import super_script as su
#hm is how many points to make
def create_data(hm,vari,step=3,correl=False):
    val=50
    ys=[]
    for i in range(hm):
        ys.append(val+random.randrange(-vari,+vari))
        if correl and correl =='pos':
            val+=step
        elif correl and correl =='neg':
            val-=step
        xs=[i for i in range(len(ys))]
    return np.array([[xs[i],ys[i]] for i in range(len(xs))])

def curveX(x,order=1):
    y=np.zeros((len(x),order+1))
    for i in range(len(x)):
        for j in range(order+1):
            y[i][j]=x[i]**j
    return y

def sq_error(y_orig,y_line):
    return sum((y_orig-y_line)**2)

def determination(y_orig,y_line):
    mean_line=[mean(y_orig) for y in y_orig]
    squared_error_regr = sum((y_orig-y_line)**2)
    squared_error_y_mean = sum((y_orig-mean_line)**2)
    return 1 - (squared_error_regr/squared_error_y_mean)




#ENTER INPUTS HERE
#ip=np.array(sorted([[10,95],[9,80],[2,10],[15,50],[10,45],[16,98],[11,38],[16,93]]))
ip=create_data(10,9,10)
accuracy=30
#separating coordinates
x=[j[0] for j in ip]
y=np.array([i[1] for i in ip]).reshape(len(ip),1)




#Full Processing
order=0

if order==0:
    print('AUTO-ORDER MODE')
    while True:
        #creating input array
        X=curveX(x,order)
        #processing
        wts=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
        pred=np.dot(X,wts)
        err=(pred-y)**2
        regr=np.dot(curveX(x,order),wts).reshape(len(x),).tolist()
        acc=determination(y.reshape(len(y),),regr)*100
        if acc>accuracy:
            break
        elif order>20:
            warnings.warn('Order greater than 20')
            break
        else:
            order+=1
        print('\norder:',order)
else:
    #creating input array
    X=curveX(x,order)
    #processing
    wts=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    pred=np.dot(X,wts)
    err=(pred-y)**2
#output
print('\nweights:\n',wts)
print('\nprediction:\n',pred)
print('\nerrors:\n',err)
regr=np.dot(curveX(x,order),wts).reshape(len(x),).tolist()
print('\nAccuracy:',acc)



#plotting
xcount=range(x[0],x[-1]+1)

label=''
k=order
plt.xlabel('x')
plt.ylabel('y') 
plt.scatter(x,y, s = 30, c = 'b')
for i in range(len(wts)):
    if k!=1 and k!=0:
        pows=su(k)
    else:
        pows=''
    if wts[i][0]>0:
        label+='+'
    label+=str('{0:.2f}'.format(wts[i][0]))+'x'+pows
    k-=1
label=label[1:]
label=label[:-1]
plt.plot(xcount,np.dot(curveX(xcount,order),wts),label = label)
plt.grid(alpha=.4,linestyle='--')
plt.legend(loc = 'best')  
plt.show()

