from numpy import loadtxt,zeros,ones,array,linspace,logspace
from pylab import scatter,show,title,xlabel,ylabel,plot,contour
def compute_cost(x,y,theta,m):
    predicts=x.dot(theta).flatten()
    error=(predicts-y)**2
    cost=(error.sum())*(1.0/(2.0*m))
    return cost