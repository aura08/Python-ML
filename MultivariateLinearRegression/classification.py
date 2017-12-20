# -*- coding: utf-8 -*-
from numpy import loadtxt,zeros,append,floor,savetxt
from Gradient import gradientDescend
alpha=0.001 #learning rate
iteration=10000  #iteratıonn of thetas update function

x=loadtxt('Test',delimiter=',',usecols=(0,1,2,3))#take xValues from Test set
y=loadtxt('Test',delimiter=',',usecols=(4))#take results of traning sets.
query=loadtxt('Query',delimiter=',',usecols=(0,1,2,3))#take query set for executing in hypothesis function.
theta=zeros(shape=(5,1))#thetas with initial values '0'

attirbutesOfHypotesis = (append([[1 for _ in range(0,len(x))]], x.T,0).T)#add as a column into x  '1' bacause of x0 values.
attirbutesOfQuery= (append([[1 for _ in range(0,len(query))]], query.T,0).T)

theta=gradientDescend(attirbutesOfHypotesis,y,theta,iteration,alpha)
result= floor(attirbutesOfQuery.dot(theta)).flatten()
query=(append(query.T,[[result.flatten()[i] for i in range(0,len(query))]],0).T)
savetxt('Out',query,fmt='%.f',delimiter=',')
print query,theta

