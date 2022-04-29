# -*- coding: UTF-8 -*-
import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
import pandas as pd
import numpy as np  
np.set_printoptions(threshold=np.inf) 
import xgboost as xgb
from sklearn import metrics,linear_model
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


tf.set_random_seed(1)

def find(a,b):
   L=len(y_p)
   sum=0
   i=0
   if len(y_t)!=L:
      print("error!")
   else:
      for i in range(L):
         if y_p[i]==a and y_t[i]==b:
            sum+=1
   return sum 


def softmax(train_batch_x,train_batch_y,test_x,test_y):
    train=np.concatenate((train_batch_x,train_batch_y),axis=1)
    train=pd.DataFrame(train,columns=COLUMNS)
    train_1=train[train.essential1==1]   
    train_0=train[train.essential1==0]  
    
    x = tf.placeholder("float", [None, 26]) 
    y_ = tf.placeholder("float", [None,2])
    W = tf.Variable(tf.zeros([26,2])) 
    b = tf.Variable(tf.zeros([2])) 
    #dense=tf.layers.dense(inputs=x, units=80, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    #x=1/(1+tf.log(-x))
    y = tf.nn.softmax(tf.matmul(x,W) + b)
        
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
        
    init = tf.global_variables_initializer()
    sess=tf.InteractiveSession()  
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    for j in range(1,10001): 
        train_set0=train_0.sample(n=750,replace='True')
        train_set1=train_1.sample(n=250,replace='True')
        train_s=pd.concat([train_set0,train_set1])
        train_set=train_s.sort_index()
        train_batch_x=train_set[FEATURES]
        train_batch_y=train_set[LABELS] 
        
        sess.run(train_step, feed_dict={x: train_batch_x, y_: train_batch_y}) 
        
    train_accuracy = accuracy.eval(feed_dict={x: train_batch_x, y_: train_batch_y})    
    
    numbers_y,numbers_y_=sess.run([y,tf.argmax(y_,1)],feed_dict={x: test_x,y_:test_y})
    np.savetxt("y_p.txt",numbers_y,delimiter=",") 
    np.savetxt("y_t.txt",numbers_y_,delimiter=",") 
    pro=np.loadtxt("y_p.txt",delimiter=",")
    t_y=np.loadtxt("y_t.txt",delimiter=",")
    t_y=t_y.reshape(t_y.shape[0],1)
    R=pro[:,1]
    pro=pro[:,1:2]
    pre=np.zeros(shape=(0,1))
    for i in range(R.shape[0]):
        if R[i]>0.5:
            y=1
        else:
            y=0
        pre=np.append(pre,y)
    y_p=pre.reshape(R.shape[0],1)
    b=1-y_p
    y_p=np.concatenate((b,y_p),axis=1)
    
    acc=metrics.accuracy_score(test_y[:,1],pre)  
    print "Softmax:",acc
    
    E=np.mean(pro)
    #print "E",E
    sigma=np.var(pro)
    #print "sigma",sigma
    error=pro-E
    all=np.concatenate((error,test_x,test_y),axis=1)
    
    return train_accuracy,pre,y_p,pro,all,sigma
    

def xgboost(train_x,train_y,test_x,test_y):
    train_y=train_y[:,1]
       
    clf =XGBClassifier(silent=True, objective='binary:logistic',min_child_weight=31)
    

    param_test = {
    'learning_rate': [0.0258,0.0259,0.026,0.0261,0.0262,0.0265,0.028],
    'max_depth':[2,3,4,5],
    'reg_lambda':[21,30,31,40,45],
    'n_estimators':[189,190,200,225],
    }
    
    '''param_test = {
    'learning_rate': [0.0256,0.0258,0.026,0.0261,0.0262,0.0264,0.028],
    'max_depth':[2,3,4,5],
    'reg_lambda':[17,20,21,30],
    'n_estimators':[21,189,190,200,210,225],
    }'''
    
    grid_search = GridSearchCV(estimator = clf, param_grid = param_test, scoring='roc_auc', cv=5)
    grid_search.fit(train_x, train_y)
    #print grid_search.best_params_ 

    y=grid_search.predict_proba(test_x)
    ypred=y[:,1]
    ypred1=y[:,1:2]  
    
    y_pred=(ypred > 0.5)*1   
    
    y_p=y_pred.reshape(y_pred.shape[0],1)
    b=1-y_p
    y_p=np.concatenate((b,y_p),axis=1)
    
    acc=metrics.accuracy_score(test_y[:,1],y_pred)  
    print "XGBOOST:",acc
    
    E=np.mean(ypred1)
    #print "E",E
    sigma=np.var(ypred1)
    #print "sigma",sigma
    error=ypred1-E
    all=np.concatenate((error,test_x,test_y),axis=1)
    
    return acc,y_pred,y_p,ypred1,all,sigma


def RF(train_x,train_y,test_x,test_y):
    y=train_y[:,1]
    cls = RandomForestClassifier(n_estimators=1000)
    cls.fit(train_x,y)
    pred=cls.predict(test_x) 
    predict=cls.predict_proba(test_x) 
    
    acc=metrics.accuracy_score(test_y[:,1], pred)
    
    R=predict[:,1]
    pro=predict[:,1:2]
    pre=np.zeros(shape=(0,1))
    for i in range(R.shape[0]):
        if R[i]>0.5:
            y=1
        else:
            y=0
        pre=np.append(pre,y)
    y_p=pre.reshape(R.shape[0],1)
    b=1-y_p
    y_p=np.concatenate((b,y_p),axis=1)
    
    print "RF:",acc
    E=np.mean(pro)
    #print "E",E
    sigma=np.var(pro)
    #print "sigma",sigma
    error=pro-E
    all=np.concatenate((error,test_x,test_y),axis=1)
    
    return acc,pre,y_p,pro,all,sigma
    
    
def LR(train_x,train_y,test_x):
    regr=linear_model.LogisticRegression()
    for j in range(1,10001): 
        regr.fit(train_x,train_y)
    pre=regr.predict_proba(test_x)   
    pre=pre[:,1:2]
    
    return pre

     
def sample(train_L,train_U,test_y,i):
    acc=0
    if acc<0.5: 
        L=train_L.sample(n=1000,replace='True')
        L=L.values
        L_x=L[:,0:26]
        L_y=L[:,26:28]
        if i==0:
            acc,R,Re,pro,all,sigma=softmax(L_x,L_y,train_U,test_y)
        elif i==1:
            acc,R,Re,pro,all,sigma=xgboost(L_x,L_y,train_U,test_y)
        elif i==2:
            acc,R,Re,pro,all,sigma=RF(L_x,L_y,train_U,test_y)
        
    return L,R,Re,pro,all,sigma

def judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3):
    sigma1=2*sigma1
    sigma2=2*sigma2
    sigma3=2*sigma3
    P1=P2=P3=np.zeros(shape=(0,28))
    err1=np.abs(all1[:,0:1])
    err2=np.abs(all2[:,0:1])
    err3=np.abs(all3[:,0:1])
    #err=np.concatenate((err1,err2,err3),axis=1)
    for j in range(all1.shape[0]):                           
        if err1[j]>sigma1 and err2[j]>sigma2:
            P3=np.concatenate((P3,all1[j:j+1,1:29]))
        if err2[j]>sigma2 and err3[j]>sigma3:
            P1=np.concatenate((P1,all2[j:j+1,1:29]))
        if err1[j]>sigma1 and err3[j]>sigma3:
            P2=np.concatenate((P2,all3[j:j+1,1:29]))
    L1=np.concatenate((L1,P1))
    L2=np.concatenate((L2,P2))
    L3=np.concatenate((L3,P3))
    print "1",L1.shape
    print "2",L2.shape
    print "3",L3.shape
    L1_y=L1[:,26:28]  
    L1_x=L1[:,0:26]
    L2_y=L2[:,26:28]
    L2_x=L2[:,0:26]
    L3_y=L3[:,26:28]
    L3_x=L3[:,0:26] 
    return L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y    
    

COLUMNS = ["dcNorm","icNorm","ecNorm","scNorm","bcNorm","ccNorm","NCNorm","PecNorm","IonNorm","P&Enorm","Vacuole","Vesicles","Lysosome","Membrane","Mitochondrion","Peroxisome","Secretory pathway","Cell wall","Cytoskeleton","Endoplasmic reticulum","Golgi","Transmembrane","Cytoplasm ","Nucleus","Endosome","Extracellular","essential0","essential1"]
FEATURES = ["dcNorm","icNorm","ecNorm","scNorm","bcNorm","ccNorm","NCNorm","PecNorm","IonNorm","P&Enorm","Vacuole","Vesicles","Lysosome","Membrane","Mitochondrion","Peroxisome","Secretory pathway","Cell wall","Cytoskeleton","Endoplasmic reticulum","Golgi","Transmembrane","Cytoplasm ","Nucleus","Endosome","Extracellular"]
LABELS = ["essential0","essential1"]

data = pd.read_csv("/home/lixia/data/Alldata.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

                           
TP=[]
TN=[]
FP=[]
FN=[]
AUC=[]

data1=data[data.essential1==1]   
a=data1.iloc[:,0].size
c=a//5
data0=data[data.essential1==0]
b=data0.iloc[:,0].size
d=b//5

data1_0=data1[:c]
data1_1=data1[c:2*c]
data1_2=data1[2*c:3*c]
data1_3=data1[3*c:4*c]
data1_4=data1[4*c:a]

data0_0=data0[:d]
data0_1=data0[d:2*d]
data0_2=data0[2*d:3*d]
data0_3=data0[3*d:4*d]
data0_4=data0[4*d:b]

data_0=pd.concat([data0_0,data1_0])
data_1=pd.concat([data0_1,data1_1])
data_2=pd.concat([data0_2,data1_2])
data_3=pd.concat([data0_3,data1_3])
data_4=pd.concat([data0_4,data1_4])

n=5

for i in range(n): 
    if i%5==0:
        train_L=data_0
        train_U=pd.concat([data_1,data_2,data_3])
        test=data_4
    elif i%5==1:
        train_L=data_1
        train_U=pd.concat([data_2,data_3,data_4])
        test=data_0
    elif i%5==2:
        train_L=data_2
        train_U=pd.concat([data_0,data_3,data_4])
        test=data_1
    elif i%5==3:      
        train_L=data_3 
        train_U=pd.concat([data_0,data_1,data_4])
        test=data_2
    elif i%5==4:
        train_L=data_4
        train_U=pd.concat([data_0,data_1,data_2])
        test=data_3
    
    test=test.sort_index()
    test=test.values
    test_x=test[:,0:26]
    test_y=test[:,27:28] 
    test_y1=test[:,26:28] 
    train_U=train_U.sample(frac=1)
    train_U=train_U.values
    train_Uy=train_U[:,26:28]
    train_U=train_U[:,0:26]
    print i
    
    for t in range(3):
        if t==0:
            L1,R1,Re1,pro1,all1,sigma1=sample(train_L,train_U[0:1018,:],train_Uy[0:1018,:],0)
            L2,R2,Re2,pro2,all2,sigma2=sample(train_L,train_U[0:1018,:],train_Uy[0:1018,:],1)
            L3,R3,Re3,pro3,all3,sigma3=sample(train_L,train_U[0:1018,:],train_Uy[0:1018,:],2)
            #er=np.concatenate((all1[:,0:1],all2[:,0:1],all3[:,0:1]),axis=1)
            #err=np.mean(er,axis=1)
            #err=err.reshape(err.shape[0],1)
            #list_=np.concatenate((err,all1[:,1:11]),axis=1)
            pro_t1=np.concatenate((pro1,pro2,pro3,train_Uy[0:1018,:]),axis=1)
            L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y=judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3) 
        elif t==1:
            acc1,R1,Re1,pro1,all1,sigma1=softmax(L1_x,L1_y,train_U[1018*t:1018*(t+1),:],train_Uy[1018*t:1018*(t+1),:])
            acc2,R2,Re2,pro2,all2,sigma2=xgboost(L2_x,L2_y,train_U[1018*t:1018*(t+1),:],train_Uy[1018*t:1018*(t+1),:])
            acc3,R3,Re3,pro3,all3,sigma3=RF(L3_x,L3_y,train_U[1018*t:1018*(t+1),:],train_Uy[1018*t:1018*(t+1),:])
            #er=np.concatenate((all1[:,0:1],all2[:,0:1],all3[:,0:1]),axis=1)
            #err=np.mean(er,axis=1)
            #err=err.reshape(err.shape[0],1)
            #list_=np.concatenate((err,all1[:,1:11]),axis=1)
            pro_t2=np.concatenate((pro1,pro2,pro3,train_Uy[1018*t:1018*(t+1),:]),axis=1)
            L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y=judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3) 
        else:
            acc1,R1,Re1,pro1,all1,sigma1=softmax(L1_x,L1_y,train_U[1018*t:,:],train_Uy[1018*t:,:])
            acc2,R2,Re2,pro2,all2,sigma2=xgboost(L2_x,L2_y,train_U[1018*t:,:],train_Uy[1018*t:,:])
            acc3,R3,Re3,pro3,all3,sigma3=RF(L3_x,L3_y,train_U[1018*t:,:],train_Uy[1018*t:,:])
            #er=np.concatenate((all1[:,0:1],all2[:,0:1],all3[:,0:1]),axis=1)
            #err=np.mean(er,axis=1)
            #err=err.reshape(err.shape[0],1)
            #list_=np.concatenate((err,all1[:,1:11]),axis=1)
            pro_t3=np.concatenate((pro1,pro2,pro3,train_Uy[1018*t:,:]),axis=1)
            L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y=judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3) 
    
    a1,r1,re1,pro1,all1,sigma1=softmax(L1_x,L1_y,test_x,test_y1)
    a2,r2,re2,pro2,all2,sigma2=xgboost(L2_x,L2_y,test_x,test_y1)
    a3,r3,re3,pro3,all3,sigma3=RF(L3_x,L3_y,test_x,test_y1)
    
    pro_t=np.concatenate((pro_t1,pro_t2,pro_t3))
    pro=np.concatenate((pro1,pro2,pro3),axis=1)
    #print pro
    #pro=pro.max(1) 
    #pro=np.mean(pro,axis=1)  
    #print pro
    pro=LR(pro_t[:,0:3],pro_t[:,4:5],pro)
    real=test_y[:,0]
    auc=metrics.roc_auc_score(real,pro)
    AUC.append(auc)
    
    pro=pro.reshape(pro.shape[0],1)
    pro=np.concatenate((pro,test_y),axis=1)
    if i==0:
        pre=pro
    else:
        pre=np.vstack((pre,pro))
        
test_1=np.ones((1167,1))  #1024
test_0=np.zeros((3926,1))   #4069
test_l=np.concatenate((test_1,test_0))
sort=pre[np.lexsort(-pre[:,::-1].T)]
sort=np.concatenate((sort,test_l),axis=1)
y_p=sort[:,2]
y_t=sort[:,1]
print sum(y_t)
  
TP=find(1.0,1.0)
TN=find(0.0,0.0)
FP=find(1.0,0.0)
FN=find(0.0,1.0)

print "AUC:",AUC
AUC=reduce(lambda x,y:x+y,AUC)/float(n)
print "AUC=",AUC
print "TP=",TP 
print "TN=",TN
print "FP=",FP
print "FN=",FN

SN=TP/float(TP+FN)
SP=TN/float(TN+FP)
FPR=FP/float(FP+TN)
PPV=TP/float(TP+FP)
NPV=TN/float(TN+FN)
F=(2*TP)/float((2*TP)+FP+FN)
ACC=(TP+TN)/float(TP+TN+FP+FN)
MCC=(TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

print "SN\t\tSP\t\tFPR\t\tPPV\t\tNPV\t\tF\t\tACC\t\tMCC"
print SN,"\t",SP,"\t",FPR,"\t",PPV,"\t",NPV,"\t",F,"\t",ACC,"\t",MCC