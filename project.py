# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:45:55 2021

@author: JK
"""
import numpy as np
from csv import reader
from random import shuffle
from random import randrange
from random import seed

## sigmoid activation function
def sigmoid(z):
    return np.divide(1,(1+np.exp(-z)))

## Derivative of sigmoid activation function
def derivative(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))
  
## cost function
def cost(X,y,theta1,theta2,labels,lamda):
    m=X.shape[0]
    grad=0;
    # calculating Cost
    J=0;
    for i in range(m):
        vec_y=np.zeros([1,labels])
        vec_y[0,y[i]]=1;
        [_,z,_,_]=propagate(X[i,:],theta1,theta2);
        J+=-1*np.sum(np.multiply(vec_y,np.log(z))+np.multiply((1-vec_y),np.log(1-z)))               
    J=J/m;
    ## Regularize cost function
    J+=lamda*(np.sum(np.square(theta1[1:,:]))+np.sum(np.square(theta2[1:,:])))/(2*m)
    
    return J

## Forward propagate
def propagate(X,theta1,theta2):
    z1=np.dot(X,theta1)
    a1=sigmoid(z1)
    a1=np.append(1,a1);
    z2=np.dot(a1,theta2)
    a2=sigmoid(z2)
    return a1,a2,z1,z2

    
## For make prediction
def predict(X,theta1,theta2):
    m=X.shape[0]
    prediction=[]
    for i in range(m):
        [a2,a3,z2,z3]=propagate(X[i,:],theta1,theta2)
        prediction.append(np.argmax(a3))
        
    return prediction

## Cross validation and accuracy 
def accuracy(actual,predicted):
    return (np.sum(actual==predicted)/len(actual))*100
      
## train network
     
# load data
fileName='seeds_dataset.csv'  ## edit your file name here
X=[];
y=[];
with open(fileName) as file_csv:
    file=reader(file_csv,delimiter=',');
    for row in file:
        temp=list(map(float,row[0:-1]));
        temp.insert(0,1)
        y.append(int(row[-1]));
        X.append(temp);
X=np.array(X)
y=np.array(y)
y=y-1;

# Noramalize data
minm=np.min(X[:,1:],0)
maxm=np.max(X[:,1:],0)
X[:,1:]=np.divide((X[:,1:]-minm),(maxm-minm));

## set the parameters
alpha=0.03;
n_hidden=5;
epoch=8600;
lamda=1;        
 
## Initialize Weights 
n_input=X.shape[1]
n_outputs=np.max(y)-np.min(y)+1;
np.random.seed(6); ## Weight matrix will be same for each run
theta1=np.random.rand(n_input,n_hidden); #Weight matrix from input layer to hidden layer
theta2=np.random.rand(n_hidden+1,n_outputs); # Weigth matrix from hidden layer to output layer

## Splitting Data into train and test 80% for training and 20% for testing
m=len(y)
test_X=[]
test_y=[]
seed(6)
for i in range(int(m*.2)):
    temp=randrange(0,m-i);
    test_X.append(X[temp,:])
    X=np.delete(X,temp,axis=0)
    test_y.append(y[temp])
    y=np.delete(y,temp)
    
train_X=X;
train_y=y;
test_y=np.array(test_y)
test_X=np.array(test_X)
    

## Training Network
m=len(train_y)
iterr=list(range(m))
shuffle(iterr)
for i in range(epoch):
    theta1_grad=np.zeros(theta1.shape)
    theta2_grad=np.zeros(theta2.shape)
    for j in iterr:
        [a2,a3,z2,z3]=propagate(train_X[j,:],theta1,theta2);
        a1=train_X[j,:]
        vec_y=np.zeros(n_outputs)
        vec_y[train_y[j]]=1
        delta3=(a3-vec_y)   ## calculate error at output neurons
        delta3=np.reshape(delta3,(len(delta3),1))
        delta2=np.dot(theta2,delta3) ## backpropagating to hidden layer step1
        delta2=delta2[1:]
        g=derivative(z2)
        g=np.reshape(g,(len(g),1)) 
        delta2=delta2*g      ## backpropagating to hidden layer step 2
        theta2_grad+=np.outer(a2,delta3)
        theta1_grad+=np.outer(a1,delta2)

    ## regularize gradient
    theta1_grad[1:,:]=(theta1_grad[1:,:]+lamda*theta1[1:,:])/m
    theta2_grad[1:,:]=(theta2_grad[1:,:]+lamda*theta2[1:,:])/m
    ## Updating Weights
    theta1=theta1-alpha*(theta1_grad)
    theta2=theta2-alpha*(theta2_grad)
    print(cost(X,y,theta1,theta2,n_outputs,lamda))



## Testing Network Accuracy
p=predict(test_X,theta1,theta2)
acc=accuracy(p,test_y); 
print("Accuracy= {:.2f}%".format(acc))
