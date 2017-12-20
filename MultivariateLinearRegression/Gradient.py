import math
from numpy import loadtxt,zeros,ones,array,linspace,logspace,insert,append,seterr
from pylab import scatter,show,title,xlabel,ylabel,plot,contour
def gradientDescend(xValues,y,theta,iteration,alpha):
    m=len(y)
    for i in range(iteration):
        for j in  range(len(theta)):
           theta[j] = theta[j] - alpha*(computeCost(xValues,y,theta,j,m))
    return theta

def computeCost(x,y,theta,i,m):
    errors=[]
    for j in range(len(y)):
         errors.append((x[j].dot(theta)-y[j])*x[j][i])
    return ( 1.0 / (2 * float(m))) * sum(errors)

