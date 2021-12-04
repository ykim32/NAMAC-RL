# ========================================================================================
# Author: Anonymized
# Date: Sep 11, 2020
# File: Concurrent AQN for Nuclear reactor operation
#       - Evaluate an induced policy with concurrent Q-averaging (AQNc, TAQNc)
# Usage: python eval_conAQN.py -method={RQN, TQN}  # select 'RQN' for AQNc, 'TQN' for TAQNc
# ========================================================================================
# * Simplified algorithm
# Multi-Agent RL for NAMAC system
# 1. Select two best agents from the induced policies (a same reward funciton with different random seeds)
# 2. Two selected agents recommend their own optimal actions given states
# 3. Average the action value and simulate the trajectory with 

# python eval_AQN.py -method=RQN -g=0 -i=0 -f=0

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import parser
import argparse

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import pandas as pd
import numpy as np
import os
import pickle
import time
import datetime
import random
import tensorflow as tf
import TQN as cq
import lib_dqn_lstm as ld


# Set the mean of original key features for vidualization & discrete action features
def initData(traindf, testdf):
    sEnv = ld.SimEnvironment()
    if polTDmode:
        sEnv.polFeat = feat + ['TD']
    
    sEnv.ps2_mean = traindf.PS2_org.mean()
    sEnv.ps2_std = traindf.PS2_org.std()
    sEnv.ct_mean = traindf.TA21s1_org.mean()
    sEnv.ct_std = traindf.TA21s1_org.std()
    sEnv.cv_mean = traindf.cv42C_org.mean() # core power generation : 61773.3 w
    sEnv.cv_std = traindf.cv42C_org.std()

    sEnv.actFeat = pd.get_dummies(traindf['Action'],prefix='a').columns.tolist()
    sEnv.outputNum = len(sEnv.simFeat)
    sEnv.n_features = len(sEnv.simFeat)
    return sEnv #, alldf


# Check the hazard of key features 
# TL14, TL8, TL9, FL1, FL6, cv42C, PS1, PS2 (+TA21s1)
# input: simulated states with original scale
def getHighTempInfo(oedf, pid, targetT):
    testNum = len(oedf[pid].unique())
    #keyFeat7 = ['TA21s1', 'cv42C', 'PS2', 'TL14', 'TL8', 'TL9', 'FL6']
    # Temperature: TA21s1
    posdf = oedf[oedf.TA21s1>targetT]
    posID = posdf.Episode.unique()
    lastT = oedf.groupby(pid).tail(1)
    
    maxHazardRate = len(posID)/testNum
    convHazardRate = len(lastT[lastT.TA21s1 > targetT]) / testNum
    hazardDur = len(posdf)*2/ testNum
    
    avgConv = lastT[keyFeat7].mean().values.round(6).tolist()
    avgMax = oedf.groupby(pid)[keyFeat7].max().mean().values.round(6).tolist()
    avgTraj = oedf.groupby(pid)[keyFeat7].mean().mean().values.round(6).tolist()     
    
    if False:
        print("* maxHazard ({}), convHazard ({}), hazardDur({}) / ConvTA21({:.2f}), MaxTA21({:.2f})/ convTL14({:.2f}), maxTL14({:.2f})".format(maxHazardRate, convHazardRate, hazardDur,  convTA21, maxTA21, convTL14, maxTL14), end=' ')
        print("Conv.power({:.2f}), Avg.power({:.2f})".format(convPower, avgPower))
    
    stateResult = [ maxHazardRate,  convHazardRate, hazardDur] + avgTraj + avgConv + avgMax
    return posID, stateResult

    
def getTargetTD(predTD):
    return np.round(predTD, 1)



def getNextState(simEnv, simulator, pdf, recAction, currentPS2, startIdx, n):

    pdf.loc[startIdx, 'a_'+str(recAction)] = 1         
    pdf.loc[startIdx, 'target_action'] = recAction

    targetPS2 = (actionMag[recAction] - simEnv.ps2_mean)/simEnv.ps2_std # standardized target PS2
    if (actionMag[recAction] == 0 ) or (currentPS2 == targetPS2): # NoAction
        incPS2 = 0
        targetPS2 = currentPS2
    else:
        incPS2 = (targetPS2 - currentPS2) / actionDur[recAction]
        if actionDur[recAction] <= 0.1:
            incPS2 = (targetPS2 - currentPS2)         
    
    maxStep = n*2 # the minimum time unit per time step is 0.5 second
    
    for j in range(maxStep):
        i = startIdx+j
        # 2.1. simulate -----------------------------------------------------------------
        tmp = pdf.loc[:i, simEnv.simFeat].values
        X = pad_sequences([tmp], maxlen = simEnv.simMaxSeqLen, dtype='float')
        yhat = simulator.predict(X, verbose=0)

        predicted = yhat[0] # [:, :len(simEnv.simFeat)]: either including TD or not, No action

        if 'TD' in simEnv.simFeat:
            predTD = predicted[0]
            tgTD = ld.getTargetTD(predTD) # approximate the next time interval: 0.5, 1, 2    
            if tgTD == 0: # after converged, set tgTD = 2
                tgTD = 2
            predicted[0] = tgTD # yhat_final = current + (predicted-current)* tgTD/predTD

            pdf.loc[i+1, 'time'] = pdf.loc[i, 'time'] + tgTD

        nextPS1_org = pdf.loc[i+1,'PS1'] # keep the original PS1 before simulation update    
        pdf.loc[i+1, simEnv.simFeat] = predicted  #*** do not update with simulated actions and TD
        pdf.loc[i+1, 'PS1'] = nextPS1_org # Replace the original PS1 weighted by TD to the simulated one
                                        # currently this period has 1-sec interval anyway. 

        # 2.2. Update next PS2 ---------------------------------------------------------------
        currentPS2 = pdf.loc[i, 'PS2'] 
            
        if (actionMag[recAction] == 0 ) or (currentPS2 == targetPS2):
            pdf.loc[i+1, 'PS2'] = currentPS2
            incPS2 = 0
        else:     
            if np.abs(currentPS2 - targetPS2) < np.abs(incPS2)* pdf.loc[i, 'TD']:
                pdf.loc[i+1, 'PS2'] = targetPS2
                incPS2 = 0
            else:
                pdf.loc[i+1, 'PS2'] = currentPS2 + incPS2 * pdf.loc[i, 'TD']#updated TD by simulation (incPS2:PS2 per second)  
        #-------------------------------------------------------------------------------------       
            
        # Check if the predicted time > n    
        if (pdf.loc[i+1, 'time'] - pdf.loc[startIdx, 'time'] > n) or (pdf.loc[i+1, 'time'] > 200):
            break

    tmp = pdf.loc[:i+1, simEnv.polFeat].values

    nextState = pad_sequences([tmp], maxlen = simEnv.polMaxSeqLen, dtype='float')  
    if DEBUG2:
        print("nextState: ========\n", nextState)
    return nextState

        
#-------------------------------------------------------------------------------------       
# Get predictive Q by averaging current Q and future Q    
#     
def getPredQ(simEnv, sess, mainQN, curState, simulator, pdf, tgTimeIdxStart, targetFutureTime, currentPS2):
    
    q_output1, recAct = simEnv.policy_sess.run([simEnv.mainQN.q_output, 
                                                simEnv.mainQN.predict], 
                                                feed_dict={simEnv.mainQN.state : curState, 
                                                simEnv.mainQN.phase : 0, 
                                                simEnv.mainQN.batch_size : 1})
    q_avg = q_output1 #np.empty([1, actionNum])    
    # 2) Get Future Q-values and average all
    targetState = curState        
    for targetT in range(len(targetFutureTime)):
        if targetFutureTime[targetT] == 0:
            continue
                
        targetState = getNextState(simEnv, simulator, pdf.copy(deep=True), recAct[0], currentPS2, tgTimeIdxStart, 
                                     n=targetFutureTime[targetT])
        q_output, recAct = simEnv.policy_sess.run([simEnv.mainQN.q_output,
                                                  simEnv.mainQN.predict],
                                                  feed_dict={simEnv.mainQN.state : targetState,
                                                            simEnv.mainQN.phase : 0, 
                                                            simEnv.mainQN.batch_size : 1})
        q_avg += q_output     
        
    if len(targetFutureTime) == 1: # only future-Q
        q_avg = q_output
    else:
	    q_avg /= len(targetFutureTime)

    return q_avg

def onlineEval_MARL(model, predf, e, tgTime, showReward):
    pdf = predf.loc[predf[pid]==e].copy(deep=True)
    pdf.reset_index(inplace=True, drop=True)
    pdf = pdf.fillna(0)
    pdf.loc[:, 'time'] = [np.round(t) for t in pdf.time.values.tolist()]
    pdf['orgTime'] = pdf['time']    
    # reset all the actions 
    pdf.loc[:, simEnv.actFeat] = 0
    pdf.loc[:, 'target_action'] = 0  # reset the actions

    maxPS2 = []
    reachMaxPS2 = 0
    tgTime.append(endTime) ## add the last time step for simulation
    initPS2 = pdf.loc[0, 'PS2'] * simEnv.ps2_std+simEnv.ps2_mean # orginal scale
    tgAct = []

    for j in range(len(tgTime)-1):
         # 2.Simulate with the recommended action up to the next action time point
        tgTimeIdxStart = pdf[pdf.time>= tgTime[j]].index[0]
        tgTimeIdxEnd = int(pdf[pdf.time<=tgTime[j+1]].index[0])  # we don't know exact next time yet 
        
        # reset the recommended action speed
        pdf.loc[tgTimeIdxStart:, 'p2speed'] = 0
        # 1. Get the recommended Action for next action, using the given policy
        tmp = pdf.loc[:tgTimeIdxStart+1, simEnv.polFeat].values

        curStates = pad_sequences([tmp], maxlen = simEnv.polMaxSeqLen, dtype='float')   
        currentPS2 = pdf.loc[tgTimeIdxStart, 'PS2']  # standardized current PS2
        
        if DEBUG2:
            print("curStates:====\n", curStates)
        #--------------------------------------------------------------------------------    
        # AQN/ TAQN : Get spatiotemporal Q-values
        q1 = getPredQ(simEnv, simEnv.policy_sess, simEnv.mainQN, curStates, simulator, pdf, tgTimeIdxStart, targetFutureTime, currentPS2)
        q2 = getPredQ(simEnv, simEnv.policy_sess2, simEnv.mainQN2, curStates, simulator, pdf, tgTimeIdxStart, targetFutureTime, currentPS2)
        q_output_avg = (q1+ q2)/2
        # q3 = getQ(simEnv, simEnv.policy_sess3, simEnv.mainQN3, curStates, simulator, pdf, tgTimeIdxStart, predN)
        # q_output_avg = (q1+ q2+ q3)/3
        #--------------------------------------------------------------------------------    
        
        
        if True:
            recAction = np.argmax(q_output_avg)
        else:
            recAction = int(np.mean([recAction1, recAction2, recAction3]))
                   
        pdf.loc[tgTimeIdxStart, 'a_'+str(recAction)] = 1         
        pdf.loc[tgTimeIdxStart, 'target_action'] = recAction

        targetPS2 = (actionMag[recAction] - simEnv.ps2_mean)/simEnv.ps2_std # standardized target PS2
        tgAct.append(recAction)
        if (actionMag[recAction] == 0 ) or (currentPS2 == targetPS2): # NoAction
            incPS2 = 0
            targetPS2 = currentPS2
        else:
            incPS2 = (targetPS2 - currentPS2) / actionDur[recAction]
        
            if actionDur[recAction] <= 0.1:
                incPS2 = (targetPS2 - currentPS2)         
               
            
        curTime = tgTime[j]
        i = tgTimeIdxStart
        
        while True:    
            # Add a line for a longer simulation than the given trajectory
            if len(pdf) == i+1:
                pdf.loc[len(pdf), :] = pdf.tail(1).values[0]    
                
            # 2.1. simulate -----------------------------------------------------------------
            tmp = pdf.loc[:i, simEnv.simFeat].values
            X = pad_sequences([tmp], maxlen = simEnv.simMaxSeqLen, dtype='float')
            yhat = simulator.predict(X, verbose=0)

            predicted = yhat[0] # [:, :len(simEnv.simFeat)]: either including TD or not (action 안들어가 어차피)
            #current = np.array(pdf.loc[i, simEnv.simFeat].values.tolist())
            
            if 'TD' in simEnv.simFeat:
                predTD = predicted[0]
                tgTD = ld.getTargetTD(predTD) # approximate the next time interval 
                if tgTD == 0: # after converged, set tgTD = 2
                    tgTD = 2
                predicted[0] = tgTD # yhat_final = current + (predicted-current)* tgTD/predTD
                
                pdf.loc[i+1, 'time'] = pdf.loc[i, 'time'] + tgTD


            nextPS1_org = pdf.loc[i+1,'PS1'] # keep the original PS1 before simulation update        
            pdf.loc[i+1,simEnv.simFeat] = predicted  #*** do not update with simulated actions and TD
            pdf.loc[i+1, 'PS1'] = nextPS1_org # Replace the original PS1 weighted by TD to the simulated one
                                              # currently this period has 1-sec interval. 
            # 2.2. Update next PS2 ---------------------------------------------------------------
            currentPS2 = pdf.loc[i, 'PS2'] 
            
            if (actionMag[recAction] == 0 ) or (currentPS2 == targetPS2):
                pdf.loc[i+1, 'PS2'] = currentPS2
                incPS2 = 0
            else:     
                if np.abs(currentPS2 - targetPS2) < np.abs(incPS2)* pdf.loc[i, 'TD']:
                    pdf.loc[i+1, 'PS2'] = targetPS2
                    incPS2 = 0
                else:  #updated TD by simulation (incPS2:PS2 per second)
                    pdf.loc[i+1, 'PS2'] = currentPS2 + incPS2 * pdf.loc[i, 'TD'] 
                                
            curTime += tgTD
            i += 1
            if curTime > tgTime[j+1]: 
                break

    pdf.drop(pdf[i:].index, inplace=True) 
    
    if showReward:
        sdf, avgRwd, simRwd, avgUtil, avgSimulUnitUtil =ld.calReward(pdf, startTime, endTime, feat, trainMean, trainStd, \
                                value_function_limits, operating_variables, 'avg', tgTimeList, key_valNorm, key_weight)
    else:
        sdf = []

    return pdf, sdf, tgAct, initPS2


    
def onlineEvalAll_byMultiRandTime_TD_p2speed(simulator, testdf, tgTimeList):
    eids = testdf.Episode.unique().tolist()
    
    #print("Multi random time with p2speed")
    if DEBUG:
        print("Total episodes ({}):".format(len(eids)), end =' ')
    
    oedf = pd.DataFrame(columns = testdf.columns)
    for i in range(len(eids)):
        if DEBUG2:
            print("\nEpisode: {}, tgTime: {}".format(eids[i], tgTimeList[i][:]))
        pdf, _, _, _ = onlineEval_MARL(simulator, testdf, eids[i], tgTimeList[i][:], showReward=False)
        oedf = pd.concat([oedf, pdf], sort=True)
    oedf.reset_index(drop=True, inplace=True)
    
    # calculate reward for the policy applied trajectories
    sdf, avgRwd, simRwd, avgUtil, avgSimulUnitUtil = ld.calReward(oedf, startTime, endTime, feat, trainMean, trainStd, \
                                      value_function_limits, operating_variables, 'avg', tgTimeList, key_valNorm, key_weight)
    
    return oedf, sdf, avgRwd, simRwd, avgUtil, avgSimulUnitUtil


def run(targetFutureTime):   
    policyEpoch, evalFile, edf = initEvalFile(evalPath, policyTitle, method, startFold, 
                     stateResultName, targetFutureTime, startIteration)
                      
    print("##### fold: {}".format(startFold))
    for pe in policyEpoch:
        epStartTime = time.time()
        print("{}".format(pe), end='\t')                                                                      
       
        for i in range(0, 1): #len(policyName)):  ## Select Policies
            sys.argv = ['',  '-g', '1', '-r', 'DR', '-a', 'elapA', '-d', str(discount[i]) , '-s', 'none',\
                    '-pb', '0', '-apx', '0', '-k', 'lstm','-c', character[i], '-l', str(0.0001), \
                    '-b', str(belief), '-hu', str(polHiddenSize),'-t',str(1),'-rp',str(1), '-msl', str(polMaxSeqLen),\
                    '-cvf', str(startFold), '-na', str(actionNum)]

            print("{}: ".format(policyTitle[0]), end='')
                
            # Load policies    
            _, simEnv.policy_sess, simEnv.mainQN, simEnv.targetQN = ld.load_policy(policyName[0], pe, polTDmode, feat)
            _, simEnv.policy_sess2, simEnv.mainQN2, simEnv.targetQN2 = ld.load_policy(policyName[1], pe, polTDmode, feat)
#             _, simEnv.policy_sess3, simEnv.mainQN3, simEnv.targetQN3 = ld.load_policy(policyName[2], pe, polTDmode, feat)

            if DEBUG2:
                print("belief: {}, targetFuture: {} sec, avgTD: {:.3f}, discount: {}".format(belief, targetFuture, avgTD,\
                                                                             belief**(avgTD/targetFuture)))
 
            testDB = testdf.copy(deep=True)
            if DEBUG:
                sEp, eEp = 23, 25
                #print("**** Episode:", sEp, eEp)
                testDB = testDB[testDB[pid].isin(testdf[pid].unique()[sEp:eEp])]


            resdf, resdf_simul2sec, avgRwd, simRwd, avgUtil, avgSimulUnitUtil= onlineEvalAll_byMultiRandTime_TD_p2speed(\
                                                                                                simulator, testDB, tgTimeList)
            posID, stateResult = getHighTempInfo(resdf_simul2sec, pid, 685)
            if DEBUG==False:
                edf.loc[len(edf)] = [simEnv.simulatorName, policyTitle[i], startFold, pe, avgSimulUnitUtil, avgUtil, avgRwd, simRwd,\
                                len(testDB)] + stateResult
                #print("evaluation time: {:.1f} min\n".format((time.time()-runStartTime)/60))
                
               
                edf.to_csv(evalFile, index=False)
        
        print("time: {:.1f} m".format((time.time()-epStartTime)/60))
    if DEBUG==False:
        print("output: ", evalFile)

def initEvalFile(evalPath, policyTitle, method, fold, stateResultName, targetFutureTime, startIteration):
    if not os.path.exists(evalPath+policyTitle[0]):
        os.makedirs(evalPath+policyTitle[0])
        
    evalFile = evalPath+policyTitle[0]+'/eval_{}{}_'.format(method, len(targetFutureTime))
    for t in targetFutureTime:
       evalFile += '{}-'.format(t)
    evalFile += 'fold{}.csv'.format(str(fold))
    
    if startIteration == 0:
        policyEpoch = [1]+[100000+i*100000 for i in range(0, 20)] 
        edf = pd.DataFrame(columns=['simulator', 'method', 'fold', 'iteration','avgSimulUnitUtil','avgUtil','avgReward',\
                                    'simReward', 'totEvents']+stateResultName) 
        print("Create a new evaluation file/: ", evalFile)

    else:
        policyEpoch = [int(args.i)+i*100000 for i in range(0, 21-int(int(args.i)/100000))] 
        edf = pd.read_csv(evalFile, header=0)
        print("Load previous evaluation file: ", evalFile)
    return policyEpoch, evalFile, edf 

# 102 action time points for Q1
# 10~100 second with at least 6 sec time interval 
tgTimeList = [[58], [34, 40, 82], [46, 58, 76], [22, 100], [16, 28, 34], [16, 46, 82], [16, 94], [46, 58], [64, 100], [10, 52], [52, 100], [16], [10, 46], [52], [40, 76], [58, 76], [28, 76], [16, 64], [94, 100], [22, 28, 70], [22, 28], [64, 94], [22, 58, 82], [10, 28], [34, 94], [58, 88], [76, 100], [34], [22, 52, 82], [28, 100], [34, 40, 64], [22], [28, 46, 70], [22, 34, 64], [28, 88], [10, 34, 88], [34, 40, 88], [16, 22, 52], [52, 82], [16, 52], [58, 100], [46], [34, 64], [10], [70], [22, 40, 100], [22, 64], [28], [88], [70, 82], [22, 64, 76], [46, 52, 88], [22, 64, 100], [46, 76], [16, 22, 34], [34, 70], [16, 40], [82], [46, 64, 88], [22, 94], [16, 76], [10, 28, 58], [28, 82], [70, 76], [94], [16, 34, 82], [70, 82, 88], [46, 88, 94], [22, 34], [64, 70], [46, 88], [82, 100], [40], [88, 100], [40, 82, 94], [52, 88], [10, 46, 70], [10, 58, 70], [22, 46, 52], [46, 52, 100], [64, 82], [64, 88, 100], [16, 28], [52, 58], [10, 100], [16, 70], [76], [28, 64], [22, 52], [64, 70, 100], [46, 52, 94], [40, 58], [10, 64], [10, 40, 64], [64], [40, 82], [58, 82], [22, 52, 94], [10, 40, 52], [22, 52, 70], [16, 34], [22, 58]]

def setAction_oneHotEncoding(testdf, actionNum):
    testdf = pd.concat([testdf, pd.get_dummies(testdf['Action'],prefix='a').fillna(int(0))], axis=1)

    actCol = ['a_'+str(i) for i in range(actionNum)]
    curCol = pd.get_dummies(testdf['Action'],prefix='a').columns.tolist()
    addCol = ['a_'+str(i) for i in range(actionNum) if 'a_'+str(i) not in curCol]
    for c in addCol:
        testdf[c] = 0

    evdf = testdf.copy(deep=True)
    if DEBUG2:
        print("setAction_oneHotEncoding - Test set: {}".format(evdf.columns))

    return evdf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-g")   # GPU id
    parser.add_argument("-f")   # fold
    parser.add_argument("-method", choices={"RQN", "TQN"})   
    parser.add_argument("-i") # start iteration
    args = parser.parse_args()
    
    # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g  # GPU-ID "0" or "0, 1" for multiple
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
#     config.gpu_options.per_process_gpu_memory_fraction = 0.1
    session = tf.Session(config=config)    

    DEBUG = True
    DEBUG2 = False
    #--------------------
    belief = 0.1
    targetFuture = 200 # seconds
    polHiddenSize = 128
    polMaxSeqLen = 5
            
    startFold = int(args.f)
    
    method = args.method
    startIteration = int(args.i)
    penalty = -1
    numConPolicies = 2
    #thresholdPenalty = False
    #--------------------
    if method == 'RQN':
        evalPath = 'eval/AQN/RQN/'
        targetFutureTime_list = [[30],[60],[120]]
        polTDmode = False  
    else:
        evalPath = 'eval/AQN/TQN/'
        targetFutureTime_list = [[30],[60],[120]]
        polTDmode = True
                
    folds = [(startFold + i)%10 for i in range(numConPolicies)]
    
    pid = 'Episode'
    timefeat = 'time'
    
    feat = ['FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C', 'cv43C'] 
        
    startTime, endTime = 10, 201 
    actionNum = 33
    actFeat = ['a_'+str(i) for i in range(actionNum)]
    feat_org = [f+'_org' for f in feat if f != 'TD']

    #actionMag, actionDur = ld.showAction(showMode = False)
    actionMag = [       0 ,  96.131027 , 97.46003 , 98.789024 , 100.11803 , 101.44703 , 102.77602 , 104.10503 , 
                105.43403 , 106.76303 , 108.09203 , 109.42103 , 110.75003 , 112.07903 , 113.40803 , 114.73703 , 
                116.06603 , 117.39503 , 118.72403 , 120.05303 , 121.38203 , 122.71103 , 124.04003 , 125.36903 , 
                126.69804 , 128.02704 , 129.35603 , 130.68503 , 132.01404 , 133.34303 , 134.67204 , 136.00104 , 137.33003 ]
    actionDur = [0]+[50]*32
    
    keras.backend.clear_session()
    date = str(datetime.datetime.now().strftime('%m%d%H'))

   
    keyFeat7 = ['TA21s1', 'cv42C', 'PS2', 'TL14', 'TL8', 'TL9', 'FL6']
    convKey7 = ['conv'+f for f in keyFeat7]
    maxKey7 = ['max'+f for f in keyFeat7]
    meanKey7 = ['mean'+f for f in keyFeat7]
    stateResultName = [ 'maxHazardRate',  'convHazardRate', 'hazardDur'] + meanKey7 + convKey7 + maxKey7
    
    targetFuture = 200
    discount = [0.9813777729719108]*2
#     fold1 = (fold + 1)%10
#     fold2 = (fold + 2)%10
    
    policyTitle = [method]
    policyPath = '../cqn/aaai21/q1_elapA/083020/'
    if 'TQN' in method: 
        character = ['LSTM_Expo']    
        methodName = '/{}_lstm_LSTM_Expo_b1_g98_h128/'.format(method)
        
    elif 'Tdiscount' in method: 
        character = ['LSTM_Expo']    
        methodName = '/{}_lstm_LSTM_Expo_b1_g98_h128/'.format(method)
        
    elif 'Tstate' in method: 
        character = ['LSTM']    
        methodName = '/{}_lstm_LSTM_b1_g98_h128/'.format(method)
        
    elif 'RQN' in method: 
        character = ['LSTM']    
        methodName = '/{}_lstm_LSTM_b1_g98_h128/'.format(method)
        
    policyName = [policyPath+str(fold)+methodName for fold in folds]
    
    
    usecols = ['Episode','time','TD']+feat+feat_org+['PS2_org_value','cv42C_org_value','TA21s1_org_value','Action', 'utility','reward']
    
    # Load data: 1_q1_elapA_Rt6c3p1_train_1215
    traindf = pd.read_csv('../data/1_q1_elapA_Rt6c3p1_train_1215.csv',header=0, usecols = usecols)
    testdf = pd.read_csv('../data/1_q1_elapA_Rt6c3p1_test_1215.csv', header=0, usecols = usecols)
    testdf = setAction_oneHotEncoding(testdf, actionNum)
    traindf['reward'] = traindf.utility.values
    testdf['reward'] = testdf.utility.values
    
    simEnv = initData(traindf, testdf)
    simEnv.simFeat = ['TD', 'FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C', 'cv43C'] 
    
    print("simFeat: ", simEnv.simFeat)
    print("polFeat: ", simEnv.polFeat)
    #### Simulator version
    simEnv.simulatorName = '1212_q1_elapA_train_msl5_h64_f15_e42'

    operating_variables = { 
        'TA21s1': {'nominal_value': 606.83, 'plusminus': 5.0, 'operating_var': True},    
        'cv42C': {'nominal_value': 62882., 'plusminus': 1.0, 'operating_var': True},
        'PS2': {'nominal_value': 91.55, 'plusminus': 16.0, 'operating_var': True},
    } 

    dataset_keys = ['time','TA21s1',  'cv42C', 'PS2']
    key_weight = np.array([0.6, 0.3, 0.1])
    value_function_limits = ld.setValueFunction(dataset_keys, operating_variables)

    #alldf = pd.concat([traindf, testdf], axis=0, sort=False)

    key_valNorm =  traindf[traindf.time>=0][['TA21s1_org_value', 'cv42C_org_value', 'PS2_org_value']].mean().values
    if DEBUG:
        print("*** Use 3 key feature reward!: {}".format(dataset_keys[1:]))
        print("key_weights: {}".format(key_weight))
        print("key_valNorm: {}, org:{}".format(key_valNorm, np.array([0.34680517, 0.45165935, 0.68064969])))
#         print("key_valNorm: {}, org:{}".format(key_valNorm, np.array([0.27875792, 0.45165935, 0.68064969])))
        
    ## Load Simulator 
    simulator = ld.load_simulator('predictor/', keyword=simEnv.simulatorName , fold='train')
    yhat = simulator.predict(pad_sequences([testdf.loc[:simEnv.simMaxSeqLen, simEnv.simFeat].fillna(0).values],\
                                                 maxlen = simEnv.simMaxSeqLen, dtype='float'), verbose=0)
   
    testvids = testdf[pid].unique().tolist()
    avgTD = traindf[traindf.TD!=0].TD.mean()
    trainMean = np.array(traindf[feat_org].mean())
    trainStd = np.array(traindf[feat_org].std())
    del traindf
    
    
    for targetFutureTime in targetFutureTime_list:
        run(targetFutureTime)
    
    
