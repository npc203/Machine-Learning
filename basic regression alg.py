from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

#hm=how many points
def create_data(hm,vari,step=3,correl=False):
    val=0
    ys=[]
    for i in range(hm):
        ys.append(val+random.randrange(-vari,+vari))
        if correl and correl =='pos':
            val+=step
        elif correl and correl =='neg':
            val-=step
        xs=[i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    
def best_fit(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    c=mean(ys)-(m*mean(xs))
    return m,c

def sq_error(y_orig,y_line):
    return sum((y_orig-y_line)**2)

def coeff_of_determination(y_orig,y_line):
    mean_line=[mean(y_orig) for y in y_orig]
    squared_error_regr = sq_error(y_orig, y_line)
    squared_error_y_mean = sq_error(y_orig, mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


xs,ys=create_data(40,10,2,'pos')
m,c= best_fit(xs,ys)
#print(m,c)
regr= [(m*x)+c for x in xs]
print(ys)
print(regr)
r_sq=coeff_of_determination(ys,regr)
print("accuracy:"+str(r_sq*100)+"%")
plt.scatter(xs,ys,color='#003F72')
plt.plot(xs,regr)
plt.show()
