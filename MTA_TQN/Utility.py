# ========================================================================================
# Author: Yeojin Kim
# Latest update: July 05, 2021
# File: UtilityEnvironment (Reactor control agent)
# ========================================================================================
import argparse
import datetime
import numpy as np
import random
import os
import time
import pickle
import pandas as pd
from collections import deque  
from scipy.stats import norm
from keras.preprocessing.sequence import pad_sequences

class UtilityEnvironment:
    DEBUG = False
    pid = 'Episode'
    timefeat = 'time'
    targetTemp = 'TA21s1' # target feature for utility in 'calReward'
    
    dataType = 'rwd3'
    feat = ['FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C', 'cv43C']
    simFeat = ['TD'] + feat[:]
    
    feat_org = [f + '_org' for f in feat]
    operating_vars = { 
            'TA21s1_org': {'nominal_value': 606.83, 'plusminus': 5, 'operating_var': True},    
            'cv42C_org': {'nominal_value': 62882., 'plusminus': 1.0, 'operating_var': True},
            'PS2_org': {'nominal_value': 91.55, 'plusminus': 17, 'operating_var': True},
        } 
    dataset_keys = ['time', 'TA21s1_org', 'cv42C_org', 'PS2_org']
    PS2feat = 'PS2_org'
    operating_keys = [k + '_value' for k in dataset_keys[1:]]
    keyFeat = ['TA21s1', 'cv42C', 'PS2'] 
    convKey = ['conv'+f for f in keyFeat]
    maxKey = ['max'+f for f in keyFeat]
    meanKey = ['mean'+f for f in keyFeat]
    stateResultName = [ 'maxHazardRate',  'convHazardRate', 'hazardDur'] + meanKey + convKey + maxKey

    startTime = 10
    endTime = 201
    normStartTime = 0
    eta = -0.05
    utilGamma = 2
    
    warningTemp = 683 
    criticalTemp = 685 
    # org: [-20, -80, -1, 100]
    # smallRwd: [-10, -40, -1, 50]
    # util25: [-5, -20, -0.5, 50] 
    # litRwd: [-5, -10, -0.5, 100]
    # rwd1s: [-10, -100, -1, 100]
    if False: #litRwd
        warningCost = -5 
        hazardCost = -10 
        hazardStayCost = -0.5
        safeReward = 100
    else: # litRwd2
        warningCost = -5 
        hazardCost = -10 
        hazardStayCost = -0.2
        safeReward = 10
    print("* Reward: warning: {}, hazard: {}, stay: {}, safe: {}".format(warningCost, hazardCost, hazardStayCost, safeReward))
    
    # Approximate an evolving failure probability based on mechanical stress
    # The baseline probability of failure (failures per s) for the pump is based on EBR II Operating docs
    failProb = 3./3./365./24./60./60.

    key_valNorm = [0.34680517, 0.45165935, 0.69675851] # init with the avg. value of operating keys in the training data 
                    #  [0.41054194 0.62172309 0.68042095]

    value_function_limits = []
    tgTimeList = [] 
    actionMag = [       0 ,  96.131027 , 97.46003 , 98.789024 , 100.11803 , 101.44703 , 102.77602 , 104.10503 , 
                105.43403 , 106.76303 , 108.09203 , 109.42103 , 110.75003 , 112.07903 , 113.40803 , 114.73703 , 
                116.06603 , 117.39503 , 118.72403 , 120.05303 , 121.38203 , 122.71103 , 124.04003 , 125.36903 , 
                126.69804 , 128.02704 , 129.35603 , 130.68503 , 132.01404 , 133.34303 , 134.67204 , 136.00104 , 137.33003 ]
    actionDur = [0]+[50]*32
    
    # Fix the order of the target Time list !!
    # 10~100 second with at least 6 sec time interval 
    # Should limit the end action time to an early stage !!!!
    # 5~60 (within 60 seconds after the accident) : startTime <= 40, 5 sec min duration.
    tgTimeList = [[10, 45, 50], [35, 45], [10, 25, 30], [20, 60], [30, 50], [5, 25, 35], [15, 20, 30], [5, 25], [35, 55, 60], 
                  [5, 20, 55], [15, 35, 50], [30, 45], [15, 40, 60], [20, 55], [35, 50], [5, 45], [5, 30, 35], [15, 20, 25],
                  [5, 10, 55], [20, 25, 40], [20, 45], [5], [10, 50, 60], [25, 55],  [10, 15, 40], [5, 50, 60], [30], [20, 25],
                  [5, 15, 50], [15, 60], [10, 25, 55], [35, 40, 55], [15, 50, 60], [20, 35, 60], [15, 25, 55], [30, 35], 
                  [15, 25], [5, 15], [20, 25, 55], [5, 30], [40], [5, 10, 50], [40, 60], [10], [20, 30], [10, 40, 50], [25, 35],
                  [15], [15, 35, 60], [5, 10, 25],  [15, 35], [5, 15, 40], [10, 35, 45], [15, 30], [10, 55], [15, 40],
                  [10, 35, 50], [20, 50], [40, 50], [5, 40], [20, 30, 35], [25, 50, 60], [35], [35, 60], [30, 55], [20, 35],
                  [25, 45], [15, 35, 45], [5, 40, 55], [15, 55], [10, 30, 35], [25], [20], [30, 60], [25, 50], [5, 10, 45],
                  [15, 25, 60], [5, 55], [10, 50], [35, 45, 60], [10, 20, 40], [35, 55], [10, 35], [15, 60], [5, 35, 40], 
                  [10, 50, 55], [15, 45, 50], [25, 30], [15, 45], [25, 35, 55],  [5, 10, 35], [20, 40], [35, 40],  [45, 60],
                  [50], [45, 50, 60], [50, 60], [50, 55], [45, 50], [45, 55]]

    def __init__(self, key_weight, traindf, method):
        self.key_weight = key_weight
        self.rewardMin = 0
        self.rewardMax = 0
        
        if  'TQN' in method or 'TState' in method: # use time interval as an input
            self.polTDmode = True 
        else: 
            self.polTDmode = False  
    
        self.avgTD =  0 #traindf[traindf.TD!=0].TD.mean()  # from training data (SARS or Org)
        self.ps2_mean = traindf.PS2_org.mean()  # from the original training data
        self.ps2_std = traindf.PS2_org.std()
        self.ct_mean = traindf.TA21s1_org.mean()
        self.ct_std = traindf.TA21s1_org.std()
        self.cv_mean = traindf.cv42C_org.mean() # core power generation : 61773.3 w
        self.cv_std = traindf.cv42C_org.std()  

        self.trainMean = np.array(traindf[self.feat_org].mean())  # from the original data
        self.trainStd = np.array(traindf[self.feat_org].std())

        self.outputNum = len(self.simFeat)
        self.n_features = len(self.simFeat)

        self.actFeat = pd.get_dummies(traindf['Action'],prefix='a').columns.tolist()
        
        _ = self.setValueFunction()
        
        
    # Set the utility for every event of the given data
    def getUtility(self, df, targetTemp):

        for i, row in self.value_function_limits.iterrows():
            x_min, x_max = row.low_limit, row.up_limit
            xVals = np.arange(x_min, x_max, 0.1)
            nominal = row.nominal
            std = row.nominal - row.low_bound

            yVals = norm.pdf(xVals,nominal,std)/max(norm.pdf(xVals,nominal,std))

            df[i+'_value'] = df[i].values
            df[i+'_value'] = df[i+'_value'].apply(lambda x: yVals[ np.argmin(np.abs(xVals-x)) ])
            df.loc[(df[i] >= x_max)|(df[i] <= x_min) ,i+'_value'] = 0

        # -----------------------------  
        # Get the utility values
        # First get the average sum of the operating variable normal distribution values
        # baseline faiure prob = external value. one of the reactor operating document. 

        if self.key_valNorm == []:
            #print("\n**Before Norm:\n {}".format(df[self.operating_keys].describe()))
            self.key_valNorm = df[df.time >=  self.normStartTime  ][self.operating_keys].mean().values
            print("** Init key_valNorm: {}".format(self.key_valNorm))
            #print("\n**After Norm:\n {}".format(df[self.operating_keys].describe()))
            
        df[self.operating_keys] = df[self.operating_keys].values / self.key_valNorm
            

        df['avg_value'] = np.mean((df[self.operating_keys].values * self.key_weight), axis=1)
        # -----------------------------

        #Get the time step - normalize failure probability per unit s
        time_step = df.groupby(self.pid).shift(-1)['time'] - df['time']  # Added

        # Maximum allowable stress is 10x the difference between nominal value and recommanded operating range (rpm)
        # !! This is an assumption !!
        max_stress = self.operating_vars[self.PS2feat]['nominal_value'] * (self.operating_vars[self.PS2feat]['plusminus']/100)
        max_stress = max_stress * 10
        if False:
            print("failure prob. = {}".format(f))
            print("Maximum allowable stress: {}".format(max_stress))

        # Get the normalized stress per time step and add the baseline failure probability
        # Cumulative sum of the two to get evolving failure probability accounting for mechanical stress
        #df['stress'] = ( np.absolute( df['PS1'].diff(periods=-1) ) / max_stress ) + (f * time_step)
        df['stress'] = ( np.absolute( df.groupby(self.pid)['PS1'].diff(periods=-1) ) / max_stress ) +\
                                (self.failProb * time_step)
        df['failure_prob'] = df['stress'].cumsum()
        df = df.drop(columns=['stress'])

        df['value'] = df['avg_value'] * np.power(df['failure_prob'], self.eta)
        df['utility'] = 1 - np.exp(-df['value']* self.utilGamma)

        # Added 
        df['utility'] = df.utility.ffill() # for the last event with NaN failure prob.
        df['utility'] = df.utility * df.groupby(self.pid).shift(1).TD.fillna(0) # !!! proportioned with Time interval
        df['reward'] = (df.utility - df.groupby(self.pid).shift(1).utility).fillna(0)

        # Penalty for safety
        # 1) Warning zone (>680) entering penalty (min: -100) / safe --> warning zone
        df.loc[(df[targetTemp] >= self.warningTemp)&(df.groupby(self.pid).shift(1)[targetTemp]<self.warningTemp),
             'reward'] += self.warningCost
        
        # 2) Hazard event(>685) entering penalty (min: -100) / warning zone --> hazard
        df.loc[(df[targetTemp] >= self.criticalTemp)&(df.groupby(self.pid).shift(1)[targetTemp]<self.criticalTemp),
             ['reward', 'utility']] += [self.hazardCost, self.hazardCost]
#                       &(df.groupby(self.pid).shift(1)[targetTemp]>=self.warningTemp),
        # 3) Hazard staying penalty
        df.loc[df[targetTemp]>=self.criticalTemp, ['reward', 'utility']] += [self.hazardStayCost, self.hazardStayCost]
        df['hazard'] = 0
        df.loc[df[targetTemp]>=self.criticalTemp, 'hazard'] = 1

        # Positive Reward for safe episode
        pep = df[df.hazard == 1][self.pid].unique().tolist()
        nep = [i for i in df[self.pid].unique().tolist() if i not in pep]
        df.loc[df[df[self.pid].isin(nep)].groupby(self.pid).tail(1).index, 'reward'] = self.safeReward
        
        if False:
            print("\tutility: {}".format(df.utility.describe().apply(lambda x: format(x, 'f'))))
            print("\treward: {}".format(df.reward.describe().apply(lambda x: format(x, 'f'))))

        #print(df[self.operating_keys+['avg_value','value','utility']].describe())

        return df       
        

    # Set the limits, bounds, nominals of key features for the value function    
    def setValueFunction(self):
        col_list = ['low_limit', 'low_bound', 'nominal', 'up_bound', 'up_limit']
        value_function_limits = pd.DataFrame(columns=col_list)

        for col in self.dataset_keys[1:]:
            if self.operating_vars[col]['operating_var']:
                low_bound = self.operating_vars[col]['nominal_value'] * (1 - (self.operating_vars[col]['plusminus']/100) )
                low_limit = self.operating_vars[col]['nominal_value'] * (1 - 3*(self.operating_vars[col]['plusminus']/100) )
                up_bound = self.operating_vars[col]['nominal_value'] * (1 + (self.operating_vars[col]['plusminus']/100) )
                up_limit = self.operating_vars[col]['nominal_value'] * (1 + 3*(self.operating_vars[col]['plusminus']/100) )
                nominal = self.operating_vars[col]['nominal_value']

                data = pd.DataFrame({ 'low_limit': [low_limit], 'low_bound': [low_bound], 'nominal': [nominal],\
                                 'up_bound': [up_bound], 'up_limit': [up_limit] }, index=[col])

                value_function_limits = value_function_limits.append(data, sort=False)
        self.value_function_limits = value_function_limits
        
        #print(self.operating_vars, self.value_function_limits)
        print("Key weights: {}".format(self.key_weight))
        return self

    # Check the hazard of key features 
    # TL14, TL8, TL9, FL1, FL6, cv42C, PS1, PS2 (+TA21s1)
    # input: simulated states with original scale
    def getHighTempInfo(self, oedf, targetT):
        testNum = len(oedf[self.pid].unique())
        #keyFeat = ['TA21s1', 'cv42C', 'PS2'] Temperature: TA21s1
        posdf = oedf[oedf.TA21s1>targetT]
        posID = posdf.Episode.unique()
        lastT = oedf.groupby(self.pid).tail(1)
    
        maxHazardRate = len(posID)/testNum
        convHazardRate = len(lastT[lastT.TA21s1 > targetT]) / testNum
        hazardDur = len(posdf)*2/ testNum
    
        avgConv = lastT[self.keyFeat].mean().values.round(6).tolist()
        avgMax = oedf.groupby(self.pid)[self.keyFeat].max().mean().values.round(6).tolist()
        avgTraj = oedf.groupby(self.pid)[self.keyFeat].mean().mean().values.round(6).tolist()     
        
        #print("(* Fuel centerline temperature - max: {}, conv: {}, avg: {})".format(maxTemp, convTemp, avgTemp))
        if False:
            print("* maxHazard ({}), convHazard ({}), hazardDur({}) / ConvTA21({:.2f}), MaxTA21({:.2f})/ convTL14({:.2f}), maxTL14({:.2f})".format(maxHazardRate, convHazardRate, hazardDur,  convTA21, maxTA21, convTL14, maxTL14), end=' ')
            print("Conv.power({:.2f}), Avg.power({:.2f})".format(convPower, avgPower))
    
        stateResult = [ maxHazardRate,  convHazardRate, hazardDur] + avgTraj + avgConv + avgMax
        return posID, stateResult

    
    
    def getTargetTD(self, predTD):
        td = [0, 0.5, 1, 2]
        a = [np.abs(t - predTD) for t in td]
        return td[a.index(min(a))]



    # make tuples (o, a, Dt, r, o', done)
    def make_tuples(self, env, df):

        obs = np.array(df[env.numFeat])
        actions = np.array(df.Action.tolist())
        next_actions = np.array(df.next_action.tolist()) 
        rewards = np.array(df[env.rewardFeat].tolist())
        done_flags = np.array(df.done.tolist())
        time_intv = np.array(df.TD.tolist()) 
        return obs, actions, time_intv, rewards, next_actions, done_flags 
 
    
    # Generate the state inputs for the validation/test data 
    def make_MT_StateData(self, env, df):
        
        zeros = np.array([0.]*len(env.numFeat))
        statePool = []
        # Init (seqLen * maxTI) size of sequence memory --------------------------------
        len_seqMEM = env.maxSeqLen * env.testMaxTI
        seqMEM = deque(maxlen=int(len_seqMEM))
        
        for i in range(len_seqMEM-1):
            seqMEM.append((zeros,0))
        # ------------------------------------------------------------------------------         
        # Make test tuples up to the time of the first action
        #print(np.shape(df), df.loc[-1*len_seqMEM:])
        obsL, actionL, tiL, rewardL, next_actionL, doneL = self.make_tuples(env, df.loc[-1*len_seqMEM:])
        
        # For each event
        for i in range(0, len(obsL)):
            seqMEM.append((obsL[i],tiL[i]))

        # Randomly select time steps for TA, given the length of seqMEM 
        sampleNum = np.min([env.maxSeqLen -1, len(seqMEM)-1])
        timeSteps = random.sample(range(0, len(seqMEM)-1), sampleNum) # select previous time steps
        timeSteps.append(len(seqMEM) - 1) # put a current time step    
        timeSteps.sort()

        # Init the working memory for temporal abstraction
        workMEM = []

        # Select (seqLen-1) number of observations and a current observation to generate a current Obs                        
        for e in range(len(timeSteps)):
            if env.TStateMode: #'TQN' in env.method or 'TState' in env.method:
                if e < len(timeSteps)-1: # for previous observations
                    accTI = 0
                    for a in range(timeSteps[e],timeSteps[e+1]-1): #Fixed: timeSteps[e+1]-1: 0302-2021
                        accTI += seqMEM[a][1]
                        #if env.UPPER_TI and (accTI >= env.UPPER_TI): # Don't abstract too sparse observations for test
                        #    break
                else: # future time interval for current observation
                    accTI = seqMEM[-1][1]    # set with the average time interval: avgTD * (env.minTrainTI+env.maxTrainTI)/2
                    #accTI = 50                # set with a fixed future time : e.g. 50
            else:
                accTI = 0
          
            cur_state, workMEM = env.makeTempAbstract(seqMEM[timeSteps[e]][0], accTI, workMEM) # Put a current observation

        return cur_state                     


    def initEval(self, env, predf, e,  tgTime):
        pdf = predf.loc[predf[self.pid]==e].copy(deep=True)
        pdf.reset_index(inplace=True, drop=True)
        pdf = pdf.fillna(0)
        pdf.loc[:, 'time'] = [np.round(t) for t in pdf.time.values.tolist()]
        pdf['orgTime'] = pdf['time']   
        
        # reset all the actions for recommendations
        pdf.loc[:, env.actFeat] = 0
        pdf.loc[:, 'target_action'] = 0  # reset the actions

        tgTime.append(self.endTime) ## add the last time step for simulation
        if self.DEBUG:
            print("tgTime: {}".format(tgTime))
        initPS2 = pdf.loc[0, 'PS2'] * self.ps2_std+self.ps2_mean # orginal scale
        
        return pdf, tgTime, initPS2 


    def simulate(self, env, pdf, simulator, tgTime, j, recAction, tgTimeIdxStart, tgAct):
        
        # Set recommended action to the trajectory
        currentPS2 = pdf.loc[tgTimeIdxStart, 'PS2']  # standardized current PS2

        pdf.loc[tgTimeIdxStart, 'a_'+str(recAction)] = 1 
        pdf.loc[tgTimeIdxStart, 'target_action'] = recAction

        targetPS2 = (self.actionMag[recAction] - self.ps2_mean)/self.ps2_std # standardized target PS2
        tgAct.append(recAction)
        if (recAction == 0 ) or (currentPS2 == targetPS2): # NoAction
            incPS2 = 0
            targetPS2 = currentPS2
        else:
            incPS2 = (targetPS2 - currentPS2) / self.actionDur[recAction]
        
            if self.actionDur[recAction] <= 0.1:
                incPS2 = (targetPS2 - currentPS2) 

        curTime = tgTime[j]
        i = tgTimeIdxStart                
        # simulate        
        while True:   
            if len(pdf) == i+1:
                pdf.loc[len(pdf), :] = pdf.tail(1).values[0] # Add a line for a longer simulation than the given trajectory
                
            # 2.1. simulate -----------------------------------------------------------------
            tmp = pdf.loc[:i, env.simEnv.simFeat].values
            X = pad_sequences([tmp], maxlen = env.simEnv.simMaxSeqLen, dtype='float')
            yhat = simulator.predict(X, verbose=0)

            predicted = yhat[0] # [:, :len(simEnv.simFeat)]: either including TD or not (action 안들어가 어차피)
            current = np.array(pdf.loc[i, env.simEnv.simFeat].values.tolist())
            
            if 'TD' in env.simEnv.simFeat:
                predTD = predicted[0]
                tgTD = self.getTargetTD(predTD) 
                if tgTD == 0: # after converged, set tgTD = 2
                    tgTD = 2
                predicted[0] = tgTD # yhat_final = current + (predicted-current)* tgTD/predTD
                
                pdf.loc[i+1, 'time'] = pdf.loc[i, 'time'] + tgTD

            nextPS1_org = pdf.loc[i+1,'PS1'] # keep the original PS1 before simulation update    
            pdf.loc[i+1,env.simEnv.simFeat] = predicted  #*** do not update with simulated actions and TD
            pdf.loc[i+1, 'PS1'] = nextPS1_org  # Replace the original PS1 weighted by TD to the simulated one
                                               # currently this period has 1-sec interval anyway. 
            # 2.2. Update next PS2 ---------------------------------------------------------------
            currentPS2 = pdf.loc[i, 'PS2'] 
            
            if (recAction == 0 ) or (currentPS2 == targetPS2):
                pdf.loc[i+1, 'PS2'] = currentPS2
                incPS2 = 0
            else:     
                if np.abs(currentPS2 - targetPS2) < np.abs(incPS2)* pdf.loc[i, 'TD']:
                    pdf.loc[i+1, 'PS2'] = targetPS2
                    incPS2 = 0
                else: #updated TD by simulation (incPS2:PS2 per second)
                    pdf.loc[i+1, 'PS2'] = currentPS2 + incPS2 * pdf.loc[i, 'TD']
                                
            curTime += tgTD
            i += 1
            if curTime > tgTime[j+1]: 
                break
        
        return pdf, tgAct, i
        
        
    def TA_action(self, env, pdf, tgTimeIdxStart):
        curStates = self.make_MT_StateData(env, pdf.loc[:tgTimeIdxStart])  #tgTimeIdxStart+1
                                
        recAction, Q = env.policySess.run([env.mainQN.predict, env.mainQN.q_output],
                                           feed_dict={env.mainQN.state : curStates,
                                                env.mainQN.phase : 0, 
                                                env.mainQN.batch_size : 1})
        #print("a:{}, Q: {}".format(recAction, Q))
        return recAction[0], Q[0]
        

    #----------------------------------------------------------------------------
    # Online evaluation with Temporal Abstraction
    
    def onlineEval_TA_episode(self, env, simulator, predf, e, tgTime, showReward=False):
    
        pdf, tgTime, initPS2 = self.initEval(env, predf, e, tgTime)
        tgAct = []
        
        for j in range(len(tgTime)-1):
            # Set the action timing and the end time
            tgTimeIdxStart = pdf[pdf.time>= tgTime[j]].index[0]
            #tgTimeIdxEnd = int(pdf[pdf.time<=tgTime[j+1]].index[0])  # we don't know exact next time yet 
        
            # Reset the recommended action speed
            pdf.loc[tgTimeIdxStart:, 'p2speed'] = 0
            # Get the recommended Action for next action, using the given policy
            recAction, _ = self.TA_action(env, pdf, tgTimeIdxStart)

            # Simulate with the recommended action up to the next action time point
            pdf, tgAct, i = self.simulate(env, pdf, simulator, tgTime, j, recAction, tgTimeIdxStart, tgAct)

        pdf.drop(pdf[i:].index, inplace=True)  

        return pdf, tgAct, initPS2    

 
            
     #----------------------------------------------------------------------------
    # Online evaluation with Multi Temporal Abstraction
    
    def onlineEval_MTA_episode(self, env, simulator, predf, e, tgTime):
    
        pdf, tgTime, initPS2 = self.initEval(env, predf, e, tgTime)
        tgAct = []
        
        for j in range(len(tgTime)-1):            
            # Set the action timing and the end time
            tgTimeIdxStart = pdf[pdf.time>= tgTime[j]].index[0]
        
            # Reset the recommended action speed
            pdf.loc[tgTimeIdxStart:, 'p2speed'] = 0
            # Get the recommended Action for next action, using the given policy
            recAct1, Q1 = self.TA_action(env, pdf, tgTimeIdxStart)
            recAct2, Q2 = self.TA_action(env, pdf, tgTimeIdxStart)
            recAct3, Q3 = self.TA_action(env, pdf, tgTimeIdxStart)
            recAct4, Q4 = self.TA_action(env, pdf, tgTimeIdxStart)
            recAct5, Q5 = self.TA_action(env, pdf, tgTimeIdxStart)
            recActs = [recAct1, recAct2, recAct3, recAct4, recAct5]
#             recActs = [recAct1, recAct2, recAct3]
            # confidence 
            C1 = Q1.max() - Q1.min()
            C2 = Q2.max() - Q2.min()
            C3 = Q3.max() - Q3.min()
            C4 = Q4.max() - Q4.min()
            C5 = Q5.max() - Q5.min()
            
#             confA = np.argmax([C1, C2, C3])
            confA = np.argmax([C1, C2, C3, C4, C5])            
            recAction = recActs[confA]
            
#             print("C1: {:.3f}({}), C2: {:.3f}({}), C3: {:.3f}({}) - recAct: {}({})".format(C1, recAct1, C2, recAct2, C3, recAct3, confA, recAction))
#             Qval = np.mean([Q1, Q2, Q3], axis=0) #Qval = np.mean([Q1, Q2, Q3, Q4, Q5], axis=0)
#             recAction = np.argmax(Qval)
            #print("meanQ: {}".format(Qval))            
            #print("recA: {} ({}, {}, {})".format(recAction, recAction1, recAction2, recAction3))

            # Simulate with the recommended action up to the next action time point
            pdf, tgAct, i = self.simulate(env, pdf, simulator, tgTime, j, recAction, tgTimeIdxStart, tgAct)

        pdf.drop(pdf[i:].index, inplace=True) 

        return pdf, tgAct, initPS2  
               
    # --------------------------------------
    
    def onlineEvalAll_TA(self, env, testdf):
        eids = testdf.Episode.unique().tolist()
        oedf = pd.DataFrame(columns = testdf.columns)
        
        if env.MTA:  # Multi-Temporal Abstraction 
            for i in range(len(eids)):
                pdf, _, _ = self.onlineEval_MTA_episode(env, env.simEnv.simulator, testdf, eids[i], self.tgTimeList[i][:])
                oedf = pd.concat([oedf, pdf], sort=True)
        else:        # Single Temporal Abstraction
            avgDurTime = []
            for i in range(len(eids)):
#                 startTime = time.time()
                pdf, _, _ = self.onlineEval_TA_episode(env, env.simEnv.simulator, testdf, eids[i], self.tgTimeList[i][:])        
                oedf = pd.concat([oedf, pdf], sort=True)
#                 durTime = time.time()-startTime
#                 avgDurTime.append(durTime)
#             print("Avg simulation time: {:.1f}".format(np.mean(avgDurTime)))
        oedf.reset_index(drop=True, inplace=True)
    
        # calculate reward 
        sdf, avgRwd, simRwd, avgUtil, avgUtil_SD, avgSimulUnitUtil = self.calReward(env.uEnv, oedf)

        return oedf, sdf, avgRwd, simRwd, avgUtil, avgUtil_SD,  avgSimulUnitUtil    

 
    # pe: policy epoch
    def getAvgUtility_TA(self, env, testdf, pe):  

        evalFile, edf = self.setEvalFile(env)
        
        # Set up the reward function ------------------------------------
        testvids = testdf[self.pid].unique().tolist()
        testDB = testdf.copy(deep=True)
            
        if self.DEBUG:
            sEp, eEp = 23,24
            testDB = testDB[testDB[self.pid].isin(testdf[self.pid].unique()[sEp:eEp])]

        # Evaluation ----------------------------------------------------
        resdf, resdf_simul2sec, avgRwd, simRwd, avgUtil, avgUtil_SD, avgUnitUtil= self.onlineEvalAll_TA(env, testDB)
        posID, stateResult = self.getHighTempInfo(resdf_simul2sec, self.criticalTemp)
        
        if self.DEBUG==False:
            edf.loc[len(edf)] = [env.simEnv.simulatorName, env.method+'_'+env.keyword, env.fold, pe+1, avgUnitUtil, avgUtil, 
                                 avgRwd, simRwd, len(testDB)] + stateResult
            edf.to_csv(evalFile, index=False)
            edf.to_csv(env.save_dir+'results/eval_fold{}.csv'.format(env.fold), index=False)
        
        return avgUtil
        
        
    def setAction_oneHotEncoding(self, testdf, actionNum):
        testdf = pd.concat([testdf, pd.get_dummies(testdf['Action'],prefix='a').fillna(int(0))], axis=1)

        actCol = ['a_'+str(i) for i in range(actionNum)]
        curCol = pd.get_dummies(testdf['Action'],prefix='a').columns.tolist()
        addCol = ['a_'+str(i) for i in range(actionNum) if 'a_'+str(i) not in curCol]
        for c in addCol:
            testdf[c] = 0

        evdf = testdf.copy(deep=True)
        if self.DEBUG:
            print("setAction_oneHotEncoding - Test set: {}".format(evdf.columns))

        return evdf


   
    def setEvalFile(self, env):
        # Evaluation file path -----------------------------------------
        evalPath = 'eval/TA/{}/lr{}_tau{}/{}/'.format(env.date, env.learnRate_init, env.tau_init, env.filename)
        if not os.path.exists(evalPath):
            os.makedirs(evalPath)
                        
        evalFile = evalPath+'eval_fold{}.csv'.format(env.fold)
        if os.path.exists(evalFile):
            edf = pd.read_csv(evalFile, header=0)
        else:
            edf = pd.DataFrame(columns=['simulator', 'method', 'fold', 'iteration','avgSimulUnitUtil','avgUtil','avgReward',\
                                    'simReward', 'totEvents']+self.stateResultName) 
        #print("Create a new evaluation file/: ", evalFile)
        return evalFile, edf

 

    def setSimpleReward(self, df, testMode):
        if testMode:
            tgFeat = 'TA21s1'
        else:
            tgFeat = 'TA21s1_org'
        lastEvents = df.groupby(self.pid).tail(1)
        df['simpleReward'] = 0

        df.loc[lastEvents[lastEvents[tgFeat] < self.criticalTemp].index, 'simpleReward'] = 10
        posEp = df[df[tgFeat] >= self.criticalTemp][self.pid].unique().tolist()
        totEp = df[self.pid].unique().tolist()
        negEp = [e for e in totEp if e not in posEp]
        negLastEvents = df[df[self.pid].isin(negEp)].groupby(self.pid).tail(1)
        df.loc[negLastEvents.index, 'simpleReward'] += 10
        return df


    def calReward(self, uEnv, pdf):
        
        sdf = pdf[(pdf.time>=uEnv.startTime) & (pdf.time<=uEnv.endTime)].copy(deep=True)
        testvids = sdf[self.pid].unique().tolist()

        # For fair evaluation, extract every 2 sec between [startTime, endTime] trajectory    
        # since each trajectory has different number of events and different time intervals
        evalTimeNum = int((uEnv.endTime - uEnv.startTime ) / 2)
        evalTimes = [uEnv.startTime+2*i for i in range(1, evalTimeNum+1)] # every 2 seconds

        sdf['time'] = (sdf.time).round(0) # round "0.5 sec unit"
        sdf.loc[sdf.time%2==1, 'time'] +=1 # convert odd second to even second

        sdf = sdf.drop_duplicates(['Episode', 'time'])
        sdf = sdf[sdf.time.isin(evalTimes)]
        sdf.reset_index(drop=True, inplace=True) 

        # calcuate reward
        sdf[self.feat]=np.array(np.array(sdf[self.feat].values.tolist())*self.trainStd+self.trainMean)
        sdf = self.getUtility(sdf, 'TA21s1')
               
        sdf = self.setSimpleReward(sdf, testMode=True)

        avgRwd =  sdf.groupby(self.pid).reward.sum().mean()
        simRwd = sdf.groupby(self.pid).simpleReward.sum().mean()
        avgUtil = sdf.groupby(self.pid).utility.sum().mean()
        avgUtil_SD = sdf.groupby(self.pid).utility.sum().std()

        ## Drop the events before the first injection of control action to calculate the average simulated unit utility
        dropIdx = []        
        for i in range(len(testvids)):
            vdf = sdf[sdf[self.pid]==testvids[i]]
            dropIdx += vdf[vdf.time < self.tgTimeList[i][0]].index.tolist()
        simuldf = sdf.copy(deep=True) 
        simuldf = simuldf.drop(dropIdx, axis=0) 
        avgSimulUnitUtil = simuldf.groupby(self.pid).utility.mean().mean()

        if False:
            for t in simuldf.time.unique().tolist():
                print("{}({})".format(np.round(t,0), len(simuldf[simuldf.time==t])), end=' ')
            print("")

        #print("util({:.2f}/{}), unitUtil({:.3f}/{})".format(avgUtil, len(sdf), avgSimulUnitUtil, len(simuldf)), end='\t')
        return sdf, avgRwd, simRwd, avgUtil, avgUtil_SD, avgSimulUnitUtil
    
    