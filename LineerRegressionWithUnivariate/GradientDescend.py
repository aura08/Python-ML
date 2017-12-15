from numpy import loadtxt,zeros,ones,array,linspace,logspace,empty
from pylab import scatter,show,title,xlabel,ylabel,plot,contour

from costFunction import compute_cost
def gradientDescend(x,y,alpha,theta,iteration,m):
  j_history=zeros(shape=(iteration,1))
  for i in range(iteration):
      predictions=x.dot(theta)
      errors_x1=sum((predictions.flatten()-y)*x[:,0])
      errors_x2=sum((predictions.flatten()-y)*x[:,1])
      theta[0][0]=(theta[0][0])-((alpha)*(1.0/float(m))*(errors_x1))
      theta[1][0]=(theta[1][0]-alpha*(1.0/float(m))*errors_x2)
      j_history[i]=compute_cost(x,y,theta,m)
  return theta,j_history