from numpy import exp, array, random, dot
def activate(x):
    return 1/(1+exp(-x))
def derv(x):
    return x * (1 - x)
X= array([[0, 0], [1, 0], [0,1]])
y= array([0, 1, 0]).T
random.seed(1)
w1 = random.random((2,4)) 
w2 = random.random((4,1)) 
#print(w1,w2,'\n\n\n')
for i in range(10000):
    z2=dot(X,w1)
    a2=activate(z2)
    z3=dot(a2,w2)
    pred=activate(z3)
    error=dot(y-pred,derv(activate(z3)))
    w2r=(0.01*dot(a2.T,error))
    w1r=0.01*dot(X.T,dot(error,w2.T)+derv(z2))
    if w1r.any()==0 or w2r.any()==0:
        print('STOPPED@:',i)
        break
    w2=w2-w2r
    w1=w1-w1r
    
print(w1,'\n\n',w2,'\n\n\n',pred)
activate(array([0,1]))

'''
for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print (1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
'''
