#!/usr/bin/env python
# coding: utf-8

"""========================================================================================
Author: Yeojin Kim
Latest update date: 07/22/2021
File: Main function for Nuclear reactor operation,
             using Temporal Abstraction with Time-aware deep Q-networks
========================================================================================
* Development environment: 
  - Linux: 3.10.0
  - Main packages: Python 3.6.9, Tensorflow 1.3.0, Keras 2.0.8, pandas 0.25.3, numpy 1.16.4
========================================================================================
* Offline data from GOTHIC simulator
========================================================================================
* Excution for each method:
DQN: $ python main.py -method=DQN, -func={Dense, LSTM}, -trainMinTI=1, -trainMaxTI=1, -testMinTI=1 -testMaxTI=8
TQN: $ python main.py -method=TQN,-func={Dense, LSTM}, -trainMinTI=1, -trainMaxTI=1, -testMinTI=1 -testMaxTI=8
TA-DQN: $ python main.py -method=TQN,-func={Dense, LSTM}, -trainMinTI=3, -trainMaxTI=5, -testMinTI=3 -testMaxTI=5
TA-TQN: $ python main.py -method=TQN,-func={Dense, LSTM}, -trainMinTI=3, -trainMaxTI=5, -testMinTI=3 -testMaxTI=5

more options:  -b 0.1 -tf 2880 -d 0.97 -hu 128 -t 200000 -msl 5 -g 3 -cvf 0 -g 0
  - b: belief for the temporal discount function
  - tf: task time window for the temporal discount function
  - d : constant discount
  - hu: number of hidden units for deep function approximation
  - t: max training update
  - msl: max sequence length for LSTM/dense
  - g: GPU ID
  - cvf: fold id for a different random seed
========================================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import pandas as pd
import argparse
import datetime
import time
import pickle
import random
from keras.preprocessing.sequence import pad_sequences

import ReactorAgent
import TQN as tq
import Utility as ut
import ReactorSimulator as rs


def round_cut(x):
    return int(np.round(x*100))/100


def get_STGamma(belief, focusTimeWindow, avgTimeInterval): # Exponential static temporal discount
    return belief**(avgTimeInterval/focusTimeWindow)

def setExponentialDiscount(env, df, belief, focusTimeWindow, avgTimeInterval, outfile):
    if 'TD' not in df.columns:
        df.loc[:, 'TD'] = (df.groupby(env.pid).shift(-1)[env.timeFeat]-df[env.timeFeat]).fillna(0).values.tolist()
        
    tgamma =  get_STGamma(belief, focusTimeWindow, avgTimeInterval) 
    
    print("static gamma: {}".format(tgamma), end='/')
    df.loc[:, env.discountFeat] = (belief**(df['TD']/focusTimeWindow)).values.tolist()
    if env.DEBUG:
        print("mean TDD (before set fillna(0)): {:.4f}".format(df[env.discountFeat].mean()), end=' ')
    #df[decayfeat] *= df.groupby(pid).shift(1)[decayfeat].fillna(1) # fillna(1): keep tgamma for the first state
    df.loc[:, env.discountFeat] = df[env.discountFeat].fillna(0).tolist() # 0 for the terminal state (at the end of trajectory)
    if env.DEBUG:
        print(" (after): {:.4f}".format(df[env.discountFeat].mean()))
    if outfile != '':
        df.to_csv(outfile, index=False)
    return df


# Discount: calculate the corresponding discount with belief and target time   
def setTemporalDiscount(env, df):   
    # Dynamic discount
    if 'TQN' in env.method or 'TDiscount' in env.method or 'Expo' in env.character:
        #print("Set the exponential discount") #: {}".format(env.discountFeat))
        df = setExponentialDiscount(env, df, env.belief, env.targetFuture, env.avgTD, outfile='')
 
    if env.DEBUG:
        print("belief: {}, targetFuture: {} sec, avgTD: {}, discount: {}".format(env.belief, env.targetFuture, env.avgTD, env.gamma)) 
        print("Episode - train: {}, val: {}".format(len(df[env.pid].unique()), len(val_df[env.pid].unique()))) 
        print("State feat: {}".format(env.stateFeat))
        
    return df


def loadPickleData(dataPath):
    with open (dataPath, 'rb') as fp:
        data = pickle.load(fp)
    return data


# For debugging
def setDEBUG(df, val_df,  env):
    #print("unique actions - train: {}, val: {}".format(df.Action.unique(), val_df.Action.unique()))
    df = df[df[env.pid].isin(df[env.pid].unique()[:20])]
    val_df = val_df[val_df[env.pid].isin(val_df[env.pid].unique()[:5])]

    env.numSteps = 10000
    env.period_eval = 2000
    env.kfold = 2
    return df, val_df, env

def setAction_oneHotEncoding(testdf, actionNum):
    testdf = pd.concat([testdf, pd.get_dummies(testdf['Action'],prefix='a').fillna(int(0))], axis=1)

    actCol = ['a_'+str(i) for i in range(actionNum)]
    curCol = pd.get_dummies(testdf['Action'],prefix='a').columns.tolist()
    addCol = ['a_'+str(i) for i in range(actionNum) if 'a_'+str(i) not in curCol]
    for c in addCol:
        testdf[c] = 0

    evdf = testdf.copy(deep=True)

    return evdf

#Make paths for our model and results to be saved in.
def createResultPaths(save_dir, date):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    if not os.path.exists(save_dir+"results"):
        os.mkdir(save_dir+"results")            
    print(save_dir)


def initData_env(df, env):    
    df = df.sort_values([env.pid, env.timeFeat])
    df.reset_index(drop=True, inplace=True)
                  
    df['done']=0
    df.loc[df.groupby(env.pid).tail(1).index, 'done'] = 1
    df['next_action'] = 0 
    df.loc[:, 'next_action'] = df.groupby(env.pid).Action.shift(-1).fillna(0)
    df['next_action'] = pd.to_numeric(df['next_action'], downcast='integer')

    # next states
    env.nextStateFeat = ['next_'+s for s in env.stateFeat]
    df[env.nextStateFeat] = df.groupby(env.pid).shift(-1).fillna(0)[env.stateFeat]
    
    # action Qs
    Qfeat = ['Q'+str(i) for i in range(len(env.actions))]
    for f in Qfeat:
        df[f] = np.nan
        
    # Temporal Difference
    df.loc[:, 'TD'] = (df.groupby(env.pid)[env.timeFeat].shift(-1) - df[env.timeFeat]).fillna(0).tolist()
    
    usecols = ['Episode', 'time'] + env.stateFeat + env.feat_org + env.nextStateFeat + env.Qfeat+['Action',  
                         'next_action', 'reward', 'utility', 'done']            
    if env.TDmode == False:
        usecols += ['TD'] 
    df = df[usecols]
    return df


def loadValData(env, val_df):
    newValpath = 'data/cv/ti{}/fold{}/'.format(env.trainMaxTI, env.cvFold)
    if ('reg' in env.keyword) or (env.testMinTI == env.testMaxTI):
        valPath = "{}/{}_{}{}_reg{}".format(newValpath, env.method, env.func_approx, env.maxSeqLen, env.testMaxTI)
    else:
        valPath = "{}/M{}_{}{}_ti{}_{}".format(newValpath, env.method, env.func_approx, env.maxSeqLen, 
                                                     env.testMinTI, env.testMaxTI)    
    if env.UPPER_TI:
        valPath += '_UPPER{}'.format(env.UPPER_TI)    
        
    if os.path.exists(valPath+'_val.p'):
        with open (valPath+'_val.p', 'rb') as fp:
            valX = np.array(pickle.load(fp))
    else:
        if not os.path.exists(newValpath):
            os.makedirs(newValpath)
        valX = env.make_MT_StateData(val_df, save_dir = newValpath)
        
    valAll = env.process_eval_batch_TA(val_df, valX) 
    return valAll
       
    
    
def loadData(SARS):
    if SARS:
        sars_train_file = 'data/2_q1_sars_w631_train_wT680_posRwd030321.csv'   
        train_file = sars_train_file             
    else:
        org_train_file = 'data/1_q1_elapA_w631_train_wT680_posRwd030321.csv'  # No sars data
        train_file = org_train_file 
    print("train file: {}".format(train_file))
    
    df = pd.read_csv(train_file, header=0) 
    val_df = pd.read_csv('data/1_q1_elapA_w631_test_wT680_posRwd030321.csv', header=0)   #3_blackout_5to10sec_train.csv
    
    adf = pd.concat([df, val_df], axis=0)
    return df, val_df, adf, org_train_file, train_file    
    

def preproc(df):
    # Set the environment
    avgTD = df[df.TD!=0].TD.mean() 
    parser = argparse.ArgumentParser()
    env = ReactorAgent.parsingPolicy(parser, avgTD, train_file) # Init a reactor environment    
    
    df = initData_env(df, env)
    
    # Cut off the training data after 'cutTime' seconds to increase training efficiency 
    if False:
        orgdfsize = len(df)
        df = df[df.time<=env.cutTime]  
        df.loc[df.groupby([env.pid]).tail(1).index, 'done'] = 1
        print("Training data: cut off after {} seconds - {} ({:.3f})".format(env.cutTime, len(df), len(df)/orgdfsize))
        print("Done flag: {}".format(len(df[df.done==1])))

    if env.DEBUG:
        df, val_df, env = setDEBUG(df, val_df,  env)

    df = tq.initPER(df, env) # For prioritized experience replay
    df = setAction_oneHotEncoding(df, env.actionNum)    
    
    #----------------------------------------------------------------
    # Set temporal discount
    df = setTemporalDiscount(env, df)
    
    return df, env, avgTD

# Validation set: baseline utility check
def getRewardVal(val_df_org):
    val_df = val_df_org.copy(deep=True)
    val_df['TA21s1'] = val_df['TA21s1_org'].values 
    val_df['PS2'] = val_df['PS2_org'].values
    val_df['cv42C'] = val_df['cv42C_org'].values  
    sdf, avgRwd, simRwd, avgUtil, avgUtil_SD, avgSimulUnitUtil = env.uEnv.calReward(env.uEnv, val_df)
    print("val_df - avg.Util: {:.1f} ({:.1f})".format(avgUtil, avgUtil_SD))    
    return avgUtil, avgUtil_SD 
    
def setReward(env, orgdf, avgTD, df, utilKeyWeight):        
    # Set the utilty class
    env.uEnv = ut.UtilityEnvironment(np.array(utilKeyWeight), orgdf, env.method)
    env.uEnv.avgTD = avgTD # with SARS data

    
    #----------------------------------------------------------------
    # Init the utility function with the statistical values of training data
    df = env.uEnv.getUtility(df, 'TA21s1_org')     
    hdf = tq.saveHyperParameters(env) 
    
    #----------------------------------------------------------------
    # For test : change the key value names
    env.uEnv.dataset_keys = ['time','TA21s1',  'cv42C', 'PS2']
    env.uEnv.operating_keys = [k + '_value' for k in env.uEnv.dataset_keys[1:]]
    env.uEnv.value_function_limits = env.uEnv.value_function_limits.rename(index={'TA21s1_org':'TA21s1', 'cv42C_org':'cv42C',
                                                                                  'PS2_org':'PS2'})
    return df

if __name__ == '__main__':
        
    date = str(datetime.datetime.now().strftime('%m/%d %H:%M'))
    print("*** [ 10-CV ] Start experiment: {}".format(date))

    # Load data
    _, _, adf, org_train_file, train_file = loadData(SARS=False)    
    # Preprocess 
    adf, env, avgTD = preproc(adf)

    # Gross validation preparation ----------------------------------------------
    # Get IDs of positive / negative hazard groups
    allIDs = adf.Episode.unique().tolist()
    if False:
        allposIDs = adf[adf.TA21s1_org>=685].Episode.unique().tolist()
        allnegIDs = [v for v in allIDs if v not in allposIDs]
        random.shuffle(allposIDs)
        random.shuffle(allnegIDs)
        with open('data/reactor_pos_ids.p', 'wb') as f:
            pickle.dump(allposIDs, f)
        with open('data/reactor_neg_ids.p', 'wb') as f:
            pickle.dump(allnegIDs, f)
    else:
        allposIDs = pickle.load(open("data/reactor_pos_ids.p", "rb" ))
        allnegIDs = pickle.load(open("data/reactor_neg_ids.p", "rb" ))
        
        
    cvPosNum = int(len(allposIDs)/10) 
    cvNegNum = int(len(allnegIDs)/10)
    print("all-pos:{} / neg: {} / all {}".format(len(allposIDs), len(allnegIDs), len(allIDs)))
    

   
    testIDcv = []
    trainIDcv = []
    for cvID in range(10): # 10-fold cv 
        testIDs = allposIDs[cvPosNum*(cvID) : cvPosNum*(cvID+1)] + allnegIDs[cvNegNum*(cvID) : cvNegNum*(cvID+1)]
        trainIDs = [v for v in allIDs if v not in testIDs]    
        testIDcv.append(testIDs)
        trainIDcv.append(trainIDs)
    # ---------------------------------------------------------------
    
    # Set up the reward function
    adf = setReward(env, adf, avgTD, adf, utilKeyWeight=[0.6, 0.3, 0.1])
    hdf = tq.saveHyperParameters(env) 
    
    #----------------------------------------------------------------
    # GPU setup 
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'          # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 		# env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto() 					# tf.compat.v1.ConfigProto()
    if True: 
        env.config.gpu_options.allow_growth = True    # Use a whole GPU
    else:  # Limite the capacity of GPU
	    env.config.gpu_options.per_process_gpu_memory_fraction = 0.1  # e.g. Use 10% of 1 GPU
    # ---------------------------------------------------------------
    session = tf.Session(config=env.config)  #tf.compat.v1.Session(config=env.config)
                      
    #----------------------------------------------------------------   
    # Training
    # tf.keras.backend.set_learning_phase(0) # commented out if needed
    # Load the simulator for online evaluation -----------------------
    env.simEnv = rs.SimEnvironment()
    # validity check of simulator
    yhat = env.simEnv.simulator.predict(pad_sequences([adf.loc[:env.simEnv.simMaxSeqLen, 
              env.simEnv.simFeat].fillna(0).values],maxlen = env.simEnv.simMaxSeqLen, dtype='float'), verbose=0)  
    
    runTime = []
    valAvgUtilList = []
    for env.cvFold in range(10):
        df = adf[adf.Episode.isin(trainIDcv[env.cvFold])].copy(deep=True)
        val_df = adf[adf.Episode.isin(testIDcv[env.cvFold])].copy(deep=True)
        testPosNum = len(val_df[val_df.TA21s1_org>=685].Episode.unique())
        print("=======================================================")
        print("Fold {} - test: pos:{}/neg:{}".format(env.cvFold, testPosNum, len(val_df.Episode.unique())-testPosNum))        
       
        valAll = loadValData(env, val_df) # Load validation data 
        
        # Experiment total 20 random seeds per method: 2 times repetitions per fold
        for i in range(env.repeat):
            if (env.cvFold == 0) and (i==0):
                print("0 - 0 : done")
                continue
            startTime = time.time()
            env.fold = env.cvFold*env.repeat + i
            env.learnRate = env.learnRate_init
            env.tau = env.tau_init
            random.seed(env.fold+1) 
            print(" ******** {}-{} *********".format(env.cvFold, env.fold) )
            
            # Generate the policy path
            env.save_dir = 'policy/{}/{}/{}/{}/'.format(env.date, env.hyperParam, env.filename, env.fold) 
            createResultPaths(env.save_dir, env.date)        

            hdf.to_csv(env.save_dir+'hyper_parameters.csv', index=False)
            print("Length: train({}), validation({})".format(len(df), len(val_df))) 

            _= tq.mt_rl_learning_sparseAction(tf, env, df, val_df.copy(deep=True), valAll) 

            curRunTime = (time.time()-startTime)/60
            runTime.append(curRunTime)
            print("Learning Time: {:.1f} hours".format(curRunTime/60))
            hdf.loc[len(hdf)] = ['learning_time', curRunTime]
            hdf.to_csv(env.save_dir+'hyper_parameters.csv', index=False)
        
          
    print("Total Learning Time: {:.1f} hours ".format(np.sum(runTime)/60))     
    print("End experiment: {}".format(str(datetime.datetime.now().strftime('%m/%d %H:%M'))))

  