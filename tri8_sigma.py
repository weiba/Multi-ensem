# -*- coding: UTF-8 -*-
import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
import pandas as pd
import numpy as np  
np.set_printoptions(threshold=np.inf) 
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics,linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


tf.set_random_seed(1)

def find(a,b,y_pre,y_true):
   L=len(y_p)
   sum=0
   i=0
   if len(y_t)!=L:
      print("error!")
   else:
      for i in range(L):
         if y_pre[i]==a and y_true[i]==b:
            sum+=1
   return sum 


def softmax(train_batch_x,train_batch_y,test_x,test_y):
    train=np.concatenate((train_batch_x,train_batch_y),axis=1)
    train=pd.DataFrame(train,columns=COLUMNS)
    train_1=train[train.essential1==1]   
    train_0=train[train.essential1==0]  
    
    x = tf.placeholder("float", [None, 8]) 
    y_ = tf.placeholder("float", [None,2])
    W = tf.Variable(tf.zeros([8,2])) 
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
        train_set0=train_0.sample(n=485,replace='True')
        train_set1=train_1.sample(n=50,replace='True')
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
    
    E=np.mean(pro)
    print "E",E
    sigma=np.var(pro)
    print "sigma",sigma
    error=pro-E
    all=np.concatenate((error,test_x,test_y),axis=1)
    
    return train_accuracy,pre,y_p,pro,all,sigma
    

def xgboost(train_x,train_y,test_x,test_y):
    train_y=train_y[:,1]
       
    clf =XGBClassifier(silent=True, objective='binary:logistic',min_child_weight=31)
    
    param_test = {
    'learning_rate': [0.0256,0.026,0.0261,0.0262,0.0264],
    'max_depth':[2,5],
    'reg_lambda':[17,20,30],
    'n_estimators':[21,200,210],
    }
    
    grid_search = GridSearchCV(estimator = clf, param_grid = param_test, scoring='roc_auc', cv=5)
    grid_search.fit(train_x, train_y)
    print grid_search.best_params_ 

    y=grid_search.predict_proba(test_x)
    ypred=y[:,1]
    ypred1=y[:,1:2]  
    
    y_pred=(ypred > 0.5)*1   
    
    y_p=y_pred.reshape(y_pred.shape[0],1)
    b=1-y_p
    y_p=np.concatenate((b,y_p),axis=1)
    
    acc=1
    
    E=np.mean(ypred1)
    print "E",E
    sigma=np.var(ypred1)
    print "sigma",sigma
    error=ypred1-E
    all=np.concatenate((error,test_x,test_y),axis=1)
    
    return acc,y_pred,y_p,ypred1,all,sigma
    

def RF(train_x,train_y,test_x,test_y):
    y=train_y[:,1]
    cls = RandomForestClassifier(n_estimators=2000)
    cls.fit(train_x,y)
    pre=cls.predict(train_x)
    acc=metrics.precision_score(y, pre)
    #predict=regr.predict(test_x)   
    predict=cls.predict_proba(test_x) 
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
    
    E=np.mean(pro)
    print "E",E
    sigma=np.var(pro)
    print "sigma",sigma
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
        L=train_L.sample(n=530,replace='True')
        L=L.values
        L_x=L[:,0:8]
        L_y=L[:,8:10]
        if i==0:
            acc,R,Re,pro,all,sigma=softmax(L_x,L_y,train_U,test_y)
        elif i==1:
            acc,R,Re,pro,all,sigma=xgboost(L_x,L_y,train_U,test_y)
        elif i==2:
            acc,R,Re,pro,all,sigma=RF(L_x,L_y,train_U,test_y)
        
    return L,R,Re,pro,all,sigma

def judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3):
    sigma1=1*sigma1
    sigma2=1*sigma2
    sigma3=1*sigma3
    P1=P2=P3=np.zeros(shape=(0,10))
    err1=np.abs(all1[:,0:1])
    err2=np.abs(all2[:,0:1])
    err3=np.abs(all3[:,0:1])
    err=np.concatenate((err1,err2,err3),axis=1)
    #print err
    for j in range(all1.shape[0]):                           
        if err1[j]>sigma1 and err2[j]>sigma2:
            P3=np.concatenate((P3,all1[j:j+1,1:11]))
        if err2[j]>sigma2 and err3[j]>sigma3:
            P1=np.concatenate((P1,all2[j:j+1,1:11]))
        if err1[j]>sigma1 and err3[j]>sigma3:
            P2=np.concatenate((P2,all3[j:j+1,1:11]))
    L1=np.concatenate((L1,P1))
    L2=np.concatenate((L2,P2))
    L3=np.concatenate((L3,P3))
    print "1",L1.shape
    print "2",L2.shape
    print "3",L3.shape
    L1_y=L1[:,8:10]  
    L1_x=L1[:,0:8]
    L2_y=L2[:,8:10]
    L2_x=L2[:,0:8]
    L3_y=L3[:,8:10]
    L3_x=L3[:,0:8] 
    return L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y    

COLUMNS = ["dc","ic","ec","sc","bc","cc","nc","ION","essential0","essential1"]
FEATURES = ["dc","ic","ec","sc","bc","cc","nc","ION"]
LABELS = ["essential0","essential1"]

data = pd.read_csv("/home/lixia/data/ECOLI.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

                           
TP=[]
TN=[]
FP=[]
FN=[]
AUC=[]
AUC1=[]
AUC2=[]
AUC3=[]

data1=data[data.essential1==1]  
data1=data1.sample(frac=1)
a=data1.iloc[:,0].size
c=a//5
data0=data[data.essential1==0]
data0=data0.sample(frac=1)
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
    test_x=test[:,0:8]
    test_y=test[:,9:10] 
    test_y1=test[:,8:10] 
    train_U=train_U.sample(frac=1)
    train_U=train_U.values
    train_Uy=train_U[:,8:10]
    train_U=train_U[:,0:8]
    
    for t in range(3):
        if t==0:
            L1,R1,Re1,pro1,all1,sigma1=sample(train_L,train_U[0:544,:],train_Uy[0:544,:],0)
            L2,R2,Re2,pro2,all2,sigma2=sample(train_L,train_U[0:544,:],train_Uy[0:544,:],1)
            L3,R3,Re3,pro3,all3,sigma3=sample(train_L,train_U[0:544,:],train_Uy[0:544,:],2)
            #er=np.concatenate((all1[:,0:1],all2[:,0:1],all3[:,0:1]),axis=1)
            #err=np.mean(er,axis=1)
            #err=err.reshape(err.shape[0],1)
            #list_=np.concatenate((err,all1[:,1:11]),axis=1)
            pro_t1=np.concatenate((pro1,pro2,pro3,train_Uy[0:544,:]),axis=1)
            L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y=judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3) 
        elif t==1:
            acc1,R1,Re1,pro1,all1,sigma1=softmax(L1_x,L1_y,train_U[544*t:544*(t+1),:],train_Uy[544*t:544*(t+1),:])
            acc2,R2,Re2,pro2,all2,sigma2=xgboost(L2_x,L2_y,train_U[544*t:544*(t+1),:],train_Uy[544*t:544*(t+1),:])
            acc3,R3,Re3,pro3,all3,sigma3=RF(L3_x,L3_y,train_U[544*t:544*(t+1),:],train_Uy[544*t:544*(t+1),:])
            #er=np.concatenate((all1[:,0:1],all2[:,0:1],all3[:,0:1]),axis=1)
            #err=np.mean(er,axis=1)
            #err=err.reshape(err.shape[0],1)
            #list_=np.concatenate((err,all1[:,1:11]),axis=1)
            pro_t2=np.concatenate((pro1,pro2,pro3,train_Uy[544*t:544*(t+1),:]),axis=1)
            L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y=judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3) 
        else:
            acc1,R1,Re1,pro1,all1,sigma1=softmax(L1_x,L1_y,train_U[544*t:,:],train_Uy[544*t:,:])
            acc2,R2,Re2,pro2,all2,sigma2=xgboost(L2_x,L2_y,train_U[544*t:,:],train_Uy[544*t:,:])
            acc3,R3,Re3,pro3,all3,sigma3=RF(L3_x,L3_y,train_U[544*t:,:],train_Uy[544*t:,:])
            #er=np.concatenate((all1[:,0:1],all2[:,0:1],all3[:,0:1]),axis=1)
            #err=np.mean(er,axis=1)
            #err=err.reshape(err.shape[0],1)
            #list_=np.concatenate((err,all1[:,1:11]),axis=1)
            pro_t3=np.concatenate((pro1,pro2,pro3,train_Uy[544*t:,:]),axis=1)
            L1,L2,L3,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y=judge(L1,L2,L3,all1,all2,all3,sigma1,sigma2,sigma3) 
    
    a1,r1,re1,pro1,all1,sigma1=softmax(L1_x,L1_y,test_x,test_y1)
    a2,r2,re2,pro2,all2,sigma2=xgboost(L2_x,L2_y,test_x,test_y1)
    a3,r3,re3,pro3,all3,sigma3=RF(L3_x,L3_y,test_x,test_y1)
    

    pro_t=np.concatenate((pro_t1,pro_t2,pro_t3))
    pro_all=np.concatenate((pro1,pro2,pro3),axis=1)
    #print pro
    #pro=pro.max(1) 
    #pro=np.mean(pro,axis=1)  
    #print pro
    pro=LR(pro_t[:,0:3],pro_t[:,4:5],pro_all)
    real=test_y[:,0]
    auc=metrics.roc_auc_score(real,pro)
    AUC.append(auc)
    
    auc1=metrics.roc_auc_score(real,pro1)
    AUC1.append(auc1)
    auc2=metrics.roc_auc_score(real,pro2)
    AUC2.append(auc2)
    auc3=metrics.roc_auc_score(real,pro3)
    AUC3.append(auc3)
    
    
    pro=pro.reshape(pro.shape[0],1)
    pro=np.concatenate((pro,test_y),axis=1)
    pro_all=np.concatenate((pro_all,test_y),axis=1)
    if i==0:
        pre=pro
        pre_all=pro_all
    else:
        pre=np.vstack((pre,pro))
        pre_all=np.vstack((pre_all,pro_all))

print pre_all.shape
pre1=np.concatenate((pre_all[:,0:1],pre_all[:,3:4]),axis=1)        
pre2=np.concatenate((pre_all[:,1:2],pre_all[:,3:4]),axis=1)    
pre3=np.concatenate((pre_all[:,2:3],pre_all[:,3:4]),axis=1)   
print pre1.shape,pre2.shape,pre3.shape     
test_1=np.ones((254,1))  
test_0=np.zeros((2473,1))   
test_l=np.concatenate((test_1,test_0))
sort=pre[np.lexsort(-pre[:,::-1].T)]
sort=np.concatenate((sort,test_l),axis=1)
y_p=sort[:,2]
y_t=sort[:,1]
#print sum(y_t)
sort1=pre[np.lexsort(-pre1[:,::-1].T)]
sort1=np.concatenate((sort1,test_l),axis=1)
y_p1=sort1[:,2]
y_t1=sort1[:,1]

sort2=pre[np.lexsort(-pre2[:,::-1].T)]
sort2=np.concatenate((sort2,test_l),axis=1)
y_p2=sort2[:,2]
y_t2=sort2[:,1]

sort3=pre[np.lexsort(-pre3[:,::-1].T)]
sort3=np.concatenate((sort3,test_l),axis=1)
y_p3=sort3[:,2]
y_t3=sort3[:,1]


all_auc=metrics.roc_auc_score(y_t,sort[:,0])
print all_auc
  
  
def result(y_pred,y_true,AUC):
    TP=find(1.0,1.0,y_pred,y_true)
    TN=find(0.0,0.0,y_pred,y_true)
    FP=find(1.0,0.0,y_pred,y_true)
    FN=find(0.0,1.0,y_pred,y_true)
    
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
    
result(y_p,y_t,AUC)
print "softmax:"
result(y_p1,y_t1,AUC1)
print "Xgboost:"
result(y_p2,y_t2,AUC2)
print "RF:"
result(y_p3,y_t3,AUC3)