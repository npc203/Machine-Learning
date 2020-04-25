from numpy import exp, array, random, dot
def act(x):
    return 1/(1+exp(-x))
def derv(x):
    x=exp(x)
    return x * (1 - x)
X= array([[0, 0], [1, 0], [0,1]])
y= array([0, 1, 0]).T
random.seed(1)
w1 = random.random((2,4)) 
w2 = random.random((4,1)) 
print(w1,'\n\n\n',w2,'\n\n\n')
for i in range(10000):
    z1=dot(X,w1)
    a1=act(z1)
    z2=dot(a1,w2)
    pred=act(z2)
    err=dot((pred-y),derv(z2))
    w2r=-dot(a1.T,err)
    w1r=-dot(X.T,derv(z1)*(w2*err.T).T) 

    if w1r.any()==0 or w2r.any()==0:
        print('STOPPED@:',i)

    w1-=w1r
    w2-=w2r
#    print('\n\n_____________\n',w1r,'\n\n\n',w2r,'\n++++++++++++')
 #   print(w1,'\n\n',w2,'\n\n\n',pred)
k=array([1,0])
z1=dot(k,w1)
print(z1.shape)
a1=act(z1)
z2=dot(a1,w2)
print('\ntrain data predicts\n',act(z2))

'''
for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print (1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
'''
