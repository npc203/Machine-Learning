from numpy import exp, array, random, dot
random.seed(1)
global w1

def act(x):
    return 1/(1+exp(-x))
def derv(x):
    return x * (1 - x)
def feed(x,w1):
    return act(dot(X,w1))
X= array([[0, 0], [1, 0], [0,1]])
y= array([0, 1, 0]).T

w1 = random.random((2,4)) 
for i in range(10000):
    out=feed(X,w1)
    print(out.shape)
    w1=w1+dot(X.T,(out-y)*derv(y))
    
print(w1,'\n\n')
activate(array([0,1]))

'''
for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print (1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
'''
