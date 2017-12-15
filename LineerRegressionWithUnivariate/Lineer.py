from numpy import loadtxt,zeros,ones,array,linspace,logspace
from pylab import scatter,show,title,xlabel,ylabel,plot,contour
from costFunction  import compute_cost
from GradientDescend import gradientDescend

data = loadtxt('exdata1mini.txt', delimiter=',') #load data values frorm txt to data variable.
x=data[:,0]#assign only first indices values of data array to x as input example
y=data[:,1]#assign only second indices values of data array to y as output example
m=y.size # number of the traning data examples.
it=ones(shape=(m,2)) # put onces to all 16*2 matrix
it[:,1]=x# put x values of it array
theta=zeros(shape=(2,1)) # pur theta values matrice like this [Q1,Q2]
iterations=10 # number of the iteration of regression
alpha=0.1# convargence value to reach optimal solution in  1000 iterations
theta,j_history=gradientDescend(it,y,alpha,theta,iterations,m)
print theta

scatter(data[:,0], data[:,1], marker='o', c='g')
a =linspace(0, 20, 1000)
b=theta[0][0]+a*theta[1][0]
plot(a,b)
show()
