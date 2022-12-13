# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 07:00:37 2022

@author: mujta
"""
import pandas as pd
import numpy as np

def prepare_dataset():
    emg_palm01 = pd.read_csv("palmch1.csv",skiprows=1, nrows=1)
    emg_palm02 = pd.read_csv("palmch2.csv",skiprows=1, nrows=1)
    emg_palm1 = pd.read_csv("palmch1.csv", nrows=1)
    emg_palm2 = pd.read_csv("palmch2.csv", nrows=1)
    emg_tip1 = pd.read_csv("tip1.csv", nrows=1)
    emg_tip2 = pd.read_csv("tip2.csv", nrows=1)
    emg_tip01 = pd.read_csv("tip1.csv", skiprows=1, nrows=1)
    emg_tip02 = pd.read_csv("tip2.csv", skiprows=1,nrows=1)
    emg_palmt1 = pd.read_csv("palmch1.csv", skiprows=2, nrows=1)
    emg_palmt2 = pd.read_csv("palmch2.csv", skiprows=2, nrows=1)
    emg_tipt1 = pd.read_csv("tip1.csv", skiprows=2, nrows=1)
    emg_tipt2 = pd.read_csv("tip2.csv", skiprows=2, nrows=1)
    emg_spherch1 = pd.read_csv("spherch1.csv", nrows=1)
    emg_spherch2 = pd.read_csv("spherch2.csv", nrows=1)
    emg_spher1 = pd.read_csv("spherch1.csv",skiprows=1, nrows=1)
    emg_spher2 = pd.read_csv("spherch2.csv",skiprows=1, nrows=1)
    emg_sphert1 = pd.read_csv("spherch1.csv", skiprows=2, nrows=1)
    emg_sphert2 = pd.read_csv("spherch2.csv", skiprows=2,nrows=1)
    
    testtemp=[0]*10*2
    testtemp1=[0]*10*2
    testtemp2=[0]*10*2
    temp=[0]*10*2
    temp1=[0]*10*2
    temp0=[0]*10*2
    temp2=[0]*10*2
    temp3=[0]*10*2
    temp4=[0]*10*2
    train = [0]*1800
    test = [0]*900
    t_testt = [0]*900
    tp = [0]*3
    tp1 = [0]*3
    tp0 = [0]*3
    tp2 = [0]*3
    tp3 = [0]*3
    t = [0]*1800
    for i in range (300):
        for j in range (10):
            temp[j]=emg_palm1.iloc[0][(i*10)+j]
            temp[j+10]=emg_palm2.iloc[0][(i*10)+j]
            if j==1:
                
                tp[j]=0
            elif j==0 :
                tp[j]=1
            elif j==2 :
                tp[j]=0    
        for j in range (10):
            temp0[j]=emg_palm01.iloc[0][(i*10)+j]
            temp0[j+10]=emg_palm02.iloc[0][(i*10)+j]
            if j==1:
                
                tp0[j]=0
            elif j==0 :
                tp0[j]=1
            elif j==2 :
                tp0[j]=0    
        for j in range (10):
            temp1[j]=emg_tip1.iloc[0][(i*10)+j]
            temp1[j+10]=emg_tip2.iloc[0][(i*10)+j]
            if j==1:
                
                tp1[j]=1
            elif j==0  :
                tp1[j]=0
            elif j==2 :
                tp1[j]=0    
        for j in range (10):
            temp2[j]=emg_tip01.iloc[0][(i*10)+j]
            temp2[j+10]=emg_tip02.iloc[0][(i*10)+j]
            if j==1:
                
                tp2[j]=1
            elif j==0 :
                tp2[j]=0 
            elif j==2 :
                tp2[j]=0    
        for j in range (10):
            temp3[j]=emg_spher1.iloc[0][(i*10)+j]
            temp3[j+10]=emg_spher2.iloc[0][(i*10)+j]
            if j==1:
                
                tp3[j]=0
            elif j==0 :
                tp3[j]=0
            elif j==2 :
                tp3[j]=1    
        for j in range (10):
            temp4[j]=emg_spherch1.iloc[0][(i*10)+j]
            temp4[j+10]=emg_spherch2.iloc[0][(i*10)+j]
            if j==1:
                
                tp3[j]=0
            elif j==0 :
                tp3[j]=0  
            elif j==2 :
                tp3[j]=1    
        train[i]=temp
        t[i]=tp
        train[300+i]=temp0
        t[300+i]=tp0
        train[600+i]=temp1
        t[600+i]=tp1
        train[900+i]=temp2
        t[900+i]=tp2
        train[1200+i]=temp3
        t[1200+i]=tp3
        train[1500+i]=temp4
        t[1500+i]=tp3
    for i in range (300):
        for j in range (10):
            testtemp[j]=emg_palmt1.iloc[0][(i*10)+j]
            testtemp[j+10]=emg_palmt2.iloc[0][(i*10)+j]
            
        for j in range (10):
            testtemp1[j]=emg_tipt1.iloc[0][(i*10)+j]
            testtemp1[j+10]=emg_tipt2.iloc[0][(i*10)+j]
            
        for j in range (10):
            testtemp2[j]=emg_sphert1.iloc[0][(i*10)+j]
            testtemp2[j+10]=emg_sphert2.iloc[0][(i*10)+j]
                 
        test[i]=testtemp
        
        test[300+i]=testtemp1
        test[600+i]=testtemp2
       
        
    for i in range (300):
        t_testt[i] = [1,0,0]
        t_testt[300+i] = [0,1,0]  
        t_testt[600+i] = [0,0,1]
    train = np.array(train)
    t = np.array(t)
    test = np.array(test)
    t_testt = np.array(t_testt)
    print(train.shape)
    print(t.shape)
    print(t_testt.shape)
    print(test.shape)
    
    return (train,t,test,t_testt)