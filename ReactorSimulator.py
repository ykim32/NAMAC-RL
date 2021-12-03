#------------------------------------------------------------------------------------------------------
# Reactor Simulator
# Author: Yeojin Kim
# Latest update: Feb 18, 2021


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import pickle

import random
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import sys
import argparse
import TQN as tq
import datetime
import time


class SimEnvironment():
    simulatorName = '1212_q1_elapA_train_msl5_h64_f15_e42'
    
    simMaxSeqLen = 5
    polMaxSeqLen = 5
    feat = ['FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C', 'cv43C'] 
    feat_org = [f+'_org' for f in feat]
    actFeat = ['a_' + str(i) for i in range(33)]

    simFeat = ['TD'] + feat[:]
    
    ps2_mean = 0
    ps2_std = 0
    ct_mean = 0
    ct_std = 0
    cv_mean = 0
    cv_std = 0
    
    outputNum = 0
    n_features = 0
    
    policy_sess = ''
    mainQN = ''
    targetQN = ''

    policy_sess2 = ''
    mainQN2 = ''
    targetQN2 = ''

    mPolicy_sess = []
    mMainQN = []
    mTargetQN = []
    
    
    def __init__(self):
        simulator = self.load_simulator('predictor/')


    def load_simulator(self, modeldir):
        modelName = modeldir+self.simulatorName+'/model_'+self.simulatorName

        # load json and create model
        json_file = open(modelName+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(modelName+".h5")
        loaded_model._make_predict_function()

        self.simulator = loaded_model
        print("Loaded simulator from disk: {}".format(modelName))

        return loaded_model

    
    
def makeXY_nStep(df, pid, inFeat, outFeat, maxSeqLen):
    date = str(datetime.datetime.now().strftime('%m/%d %H:%M'))
    print("Start making LSTM data: {}".format(date))
    startTime = time.time()
    
    IDs = df[pid].unique().tolist()
    X = []
    init = 1

    for e in IDs:
        tdf = df.loc[df[pid]==e, inFeat]
        tmp = tdf.values
        tmpX = []
        for i in range(len(tmp)):
            tmpX.append(pad_sequences([tmp[:i+1]], maxlen = maxSeqLen, dtype='float')) 

        if init == 1:
            X = tmpX
            init = 0
        else:
            X = np.concatenate((X, tmpX))
            
    Y = df.groupby(pid).shift(-1).ffill()[outFeat].values 
    X = X.reshape((X.shape[0], X.shape[2], len(inFeat)))
    print(np.shape(X), np.shape(Y))
    print("making data: {:.1f} min".format((time.time()-startTime)/60))
    return X, Y




def getSMAPE_individualFeat(model, testX, testY, inFeat, outFeat):
    res = pd.DataFrame(columns = ['feat', 'SMAPE'])

    yhat = model.predict(testX, verbose=0)
    
    for i in range(len(outFeat)):
        smape = np.sum(np.abs(testY[:,i]-yhat[:,i])/(np.abs(testY[:,i])+np.abs(yhat[:,i]))/2)/len(yhat)*100
        #print("{}\t{:.4f}".format(outFeat[i], mape))
        res.loc[len(res)] = [outFeat[i], smape]
    print("avg. MAPE over all the features: {:.4f}".format(res.SMAPE.mean()))
    return yhat, res


def getPredErrors_individualFeat(model, testX, testY, inFeat, outFeat):
    res = pd.DataFrame(columns = ['feat', 'SMAPE', 'NRMSE','NRMSA'])

    yhat = model.predict(testX, verbose=0)
    
    for i in range(len(outFeat)):
        smape = np.sum(np.abs(testY[:,i]-yhat[:,i])/(np.abs(testY[:,i])+np.abs(yhat[:,i]))/2)/len(yhat)*100
        nrmse = np.sqrt(np.sum( (testY[:,i]-yhat[:,i])**2)/len(yhat))/ (np.max(testY[:,i])-np.min(testY[:,i]))
        nrmsa = 100*(1-np.sqrt(np.sum((testY[:,i]-yhat[:,i])**2)/np.sum((testY[:,i]-np.mean(testY[:,i]))**2)))

        #print("{}\t{:.4f}".format(outFeat[i], mape))
        res.loc[len(res)] = [outFeat[i], smape, nrmse, nrmsa]
    print("All(TD)- avg. SMAPE: {:.4f} / avg. NRMSE: {:.5f} / avg. NRMSA: {:.4f}".format(res.SMAPE.mean(),res.NRMSE.mean(),\
                                                                                         res.NRMSA.mean()))
    stateRes = res[res.feat!='TD']
    exCtrlRes = res[(res.feat!='TD')&(res.feat!='PH2')&(res.feat!='PS2')]
    print("States - avg. SMAPE: {:.4f} / avg. NRMSE: {:.5f} / avg. NRMSA: {:.4f}".format(stateRes.SMAPE.mean(), \
                                                                      stateRes.NRMSE.mean(), stateRes.NRMSA.mean()))
    print("Ex Ctrl- avg. SMAPE: {:.4f} / avg. NRMSE: {:.5f} /avg. NRMSA: {:.4f}".format(exCtrlRes.SMAPE.mean(), \
                                                                      exCtrlRes.NRMSE.mean(), exCtrlRes.NRMSA.mean()))
    return yhat, res

#----------------
# RL

class PolEnvironment(object):
    # class attributes
    config = []
    
    pid = 'Episode'
    label = 'Unsafe'
    timeFeat = 'time'
    discountFeat = 'DynamicDiscount' 
    rewardFeat = 'reward' #'Reward'

    DUELING = False
    DOUBLE = False
    
    date = ''
 
    #numFeat = ['FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8s1', 'TL9s1', 'TL14s1', 'PS1', 'PS2','PH1','PH2','cv42C','cv43C']
        
    train_posvids = []
    train_negvids = [] 
    train_totvids = []
    test_posvids = []
    test_negvids = [] 
    test_totvids = []
    
     
    def __init__(self, args, numFeat, polFeat):
        self.rewardType = args.r
        self.keyword = args.k
        self.load_data = args.a
        self.character = args.c
        self.gamma = float(args.d)
        self.splitter = args.s
        self.streamNum = 0
        self.LEARNING_RATE = True # use it or not
        self.learnRate = float(args.l) # init_value (αk+1 = 0.98αk)
        self.learnRateFactor = 0.98
        self.learnRatePeriod = 5000
        self.belief = float(args.b)
        self.hidden_size = int(args.hu)
        self.numSteps = int(args.t)
        self.discountFeat = args.df
        self.pred_basis = 0
        self.gpuID = str(args.g)
        self.apx = float(args.apx)
        self.repeat = int(args.rp)
        self.maxSeqLen = int(args.msl)

        self.per_flag = True
        self.per_alpha = 0.6 # PER hyperparameter
        self.per_epsilon = 0.01 # PER hyperparameter
        self.beta_start = 0.9 # the lower, the more prioritized
        self.reg_lambda = 5
        self.Q_clipping = False # for Q-value clipping 
        self.Q_THRESHOLD = 1000 # for Q-value clipping
        self.REWARD_THRESHOLD = 1000
        self.tau = 0.001 #Rate to update target network toward primary network
        if 'pred' in self.splitter:
            env.pred_basis = float(args.pb)
            
        self.pred_res = 0 # inital target prediction result for netowkr training
        self.gamma_rate = 1 # gamma increasing rate (e.g. 1.001)
        
        self.DEBUG = False 
        self.targetTimeWindow = 0
        self.load_model = False #True
        self.save_results = True
        self.func_approx = 'LSTM' #'FC_S2' #'FC' 
        self.batch_size = 32
        self.period_save = 10000
        self.period_eval = 10000
        self.saveResultPeriod = 200000
 
        self.splitInfo = 'none'
        self.filename = self.splitter+'_'+self.keyword+'_'+self.character +'_b'+str(int(self.belief*10))+ '_g'+ \
                        str(int(self.gamma*100)) +'_h'+str(self.hidden_size)+ '_'+self.load_data 
        self.fold = int(args.cvf)
        self.numFeat = numFeat
        self.nextNumFeat = [f + '_next' for f in numFeat]
        self.stateFeat = polFeat#[]
        self.nextStateFeat = [f + '_next' for f in polFeat]
        
        self.policyName = ''
        self.actions = [i for i in range(int(args.na))] # no action, 0.8, 0.9, 1.0, 1.1, 1.2
        self.Qfeat = ['Q'+str(i) for i in self.actions]
        

        
def setGPU(tf, env):
        # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto()
    env.config.gpu_options.per_process_gpu_memory_fraction = 0.05
    return env

def parsing(parser, polTDmode, feat):   
    parser.add_argument("-a")# load_data
    parser.add_argument("-l")
    parser.add_argument("-t")    
    parser.add_argument("-g")   # GPU ID#
    parser.add_argument("-r")   # i: IR or DR
    parser.add_argument("-k")   # keyword for models & results
    parser.add_argument("-msl")  # max sequence length for LSTM
    parser.add_argument("-d")   # discount factor gamma
    parser.add_argument("-s")   # splitter: prediction
    parser.add_argument("-apx")   # sampling mode for prediction: approx or not 
    parser.add_argument("-pb") # pred_val basis to distinguish pos from neg (0.5, 0.9, etc.)
    parser.add_argument("-c") # characteristics of model
    parser.add_argument("-b") # belief for dynamic TDQN
    parser.add_argument("-hu") # hidden_size
    parser.add_argument("-df") # discount feature (Expo or Hyper)
    parser.add_argument("-rp") # repeat to build a model
    parser.add_argument("-cvf") # repeat to build a model
    parser.add_argument("-na") # number of categorical actions
    
    args = parser.parse_args()

    if polTDmode:
        polStateFeat = feat + ['TD']
    else:
        polStateFeat = feat

    polEnv = PolEnvironment(args, feat, polStateFeat)
    #env.stateFeat = env.numFeat[:]
    
    # update numfeat & state_features
       
    if 'pf' in polEnv.character: # = predFeat
        polEnv.numfeat += ['pred_val']
        polEnv.stateFeat += ['pred_val']
        
    if 'MI' in polEnv.character:
        polEnv.stateFeat += [i+'_mi' for i in polEnv.numfeat]
    
    # if pf or MI, then update nest feature
    polEnv.nextStateFeat = ['next_'+s for s in polEnv.stateFeat]
    polEnv.nextNumFeat = ['next_'+s for s in polEnv.numFeat]
    
    #print("nextNumFeat: ", env.nextNumFeat)   
    #print("nextstateFeat: ", env.nextStateFeat)

    # update filename    
    if 'pred' in polEnv.splitter:
        polEnv.filename = polEnv.filename + '_pb'+str(int(polEnv.pred_basis*1000))
        
    return polEnv
    
import os
def save_model(model, newdir, keyword):
    path = newdir+'/'+keyword+'/'
    if not os.path.exists(path):
        os.makedirs(path)
        
    # serialize model to JSON
    model_json = model.to_json()
    with open(path+"model_"+keyword+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+"model_"+keyword+".h5")
    print("Saved model: {}".format(path))
    

def load_model(file):
    # load json and create model
    json_file = open(file+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file+".h5")
    #print("Loaded model from disk")
    return loaded_model

def load_simulator(modeldir, keyword, fold):
    if fold == 'train':
        modelName = modeldir+keyword+'/model_'+keyword
    elif fold == 'all':
        modelName = modeldir+keyword+'/model_'+keyword+'_'+fold
    else:
        modelName = modeldir+keyword+'/model_'+keyword+'_cv'+str(fold)
    
    # load json and create model
    json_file = open(modelName+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelName+".h5")
    loaded_model._make_predict_function()

    print("Loaded simulator from disk: {}".format(keyword))
    
    return loaded_model


def load_policy(policy_dir, policyEpoch, polTDmode, feat):
    policyName = policy_dir+'models/pol_'+str(policyEpoch)+'/'
    #print("policy:", policyName)
    # Load the policy network
    parser = argparse.ArgumentParser()
    polEnv = parsing(parser, polTDmode, feat)
    polEnv = setGPU(tf, polEnv)
    
    if 'Clean' in policyName or 'clean' in policyName:
        polEnv.DUELING = False
        polEnv.DOUBLE = False
    elif 'noPER' in policyName:
        polEnv.DUELING = True
        polEnv.DOUBLE = True
        #print("** Dueling + Double")
    else:
        polEnv.DUELING = True
        polEnv.DOUBLE = True
        polEnv.per_flag = True
        #print("** Dueling + Double + PER")
        
    
    #print("polEnv.stateFeat:", polEnv.stateFeat)
    #mainQN, targetQN, saver, init = setNetwork(tf, polEnv)
    tf.reset_default_graph()

    mainQN = tq.RQnetwork(polEnv, 'main')
    targetQN = tq.RQnetwork(polEnv, 'target')

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    policy_sess = tf.Session(config=polEnv.config) 
    load_RLmodel(policy_sess, tf, policyName)
    
    
    return polEnv, policy_sess, mainQN, targetQN 

def setNetwork(tf, env):
    tf.reset_default_graph()

    mainQN = tq.RQnetwork(env, 'main')
    targetQN = tq.RQnetwork(env, 'target')

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    return mainQN, targetQN, saver, init    

def load_RLmodel(sess, tf, save_dir):
    startTime = time.time()
    try: # load RL model
        restorer = tf.train.import_meta_graph(save_dir + 'ckpt.meta')
        restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
        
    except IOError:
        print ("Error: No previous model found!") 
        
def setAction(df):    
    df['Action'] = 0
    cnt = 1
    for d in actDurs:
        for a in actions:
            df.loc[(df.actDur_org == d)&(df.action_org==a),'Action'] = cnt
            cnt +=1
    return df

def showAction(showMode):
    cnt = 1
    actMag = [0.8, 0.9, 1., 1.1, 1.2]
    actDur = [0.1, 15, 29, 34, 84]
    actionMag = [0]
    actionDur = [0]
    for a in actMag:
        for d in actDur:
            actionMag.append(a)
            actionDur.append(d)
            if showMode:
                print("(actionMag, actionDur)")
                print("{}:({}, {}) ".format(cnt, a, d), end=' ')
            cnt +=1
    return actionMag, actionDur


def getTargetTD(predTD):
    td = [0, 0.5, 1, 2]
    a = [np.abs(t - predTD) for t in td]
    return td[a.index(min(a))]



#----------------------------------------------------------------------------------------
# Reward function

from scipy.stats import norm

# modified / temporary #
# assuming that PS1P == PS1 and PS2P == PS2 #
# also set PS1 & PS2 nominal values to 91.55 (value at beginning of simulation)

def setValueFunction(uEnv):
    col_list = ['low_limit', 'low_bound', 'nominal', 'up_bound', 'up_limit']
    value_function_limits = pd.DataFrame(columns=col_list)

    for col in uEnv.dataset_keys[1:]:
        if uEnv.operating_variables[col]['operating_var']:

            low_bound = uEnv.operating_variables[col]['nominal_value'] * (1 - (uEnv.operating_variables[col]['plusminus']/100) )
            low_limit = uEnv.operating_variables[col]['nominal_value'] * (1 - 3*(uEnv.operating_variables[col]['plusminus']/100) )
            up_bound = uEnv.operating_variables[col]['nominal_value'] * (1 + (uEnv.operating_variables[col]['plusminus']/100) )
            up_limit = uEnv.operating_variables[col]['nominal_value'] * (1 + 3*(uEnv.operating_variables[col]['plusminus']/100) )
            nominal = uEnv.operating_variables[col]['nominal_value']

            data = pd.DataFrame({ 'low_limit': [low_limit], 'low_bound': [low_bound], 'nominal': [nominal],\
                                 'up_bound': [up_bound], 'up_limit': [up_limit] }, index=[col])

            value_function_limits = value_function_limits.append(data, sort=False)
    uEnv.value_function_limits = value_function_limits
    return uEnv
            

def getReward0923(hist000, value_function_limits, operating_variables, weightFlag, key_valNorm, key_weight):
    pid = 'Episode'
    operating_keys = []
    for i, row in value_function_limits.iterrows():
        operating_keys.append(i+'_value')
        x_min, x_max = row.low_limit, row.up_limit
        xVals = np.arange(x_min, x_max, 0.1)
        nominal = row.nominal
        std = row.nominal - row.low_bound

        yVals = norm.pdf(xVals,nominal,std)/max(norm.pdf(xVals,nominal,std))

        hist000[i+'_value'] = hist000[i].values
        hist000[i+'_value'] = hist000[i+'_value'].apply(lambda x: yVals[ np.argmin(np.abs(xVals-x)) ])
        hist000.loc[(hist000[i] >= x_max)|(hist000[i] <= x_min) ,i+'_value'] = 0
    
    # Get the utility values
    # First get the average sum of the operating variable normal distribution values
    # baseline faiure prob = external value. one of the reactor operating document. 
    hist000['avg_value'] = hist000[operating_keys].mean(axis=1)

    # Approximate an evolving failure probability based on mechanical stress
    # The baseline probability of failure (failures per s) for the pump is
    # Based on EBR II Operating docs
    f = 3./3./365./24./60./60.
    
    #Get the time step - normalize failure probability per unit s
    time_step = (hist000.groupby(pid).shift(-1)['time'] - hist000['time']).fillna(0).values  # Revised by Yeojin

    # Maximum allowable stress is 10x the difference between nominal value and recommanded operating range (rpm)
    # !! This is an assumption !!
    max_stress = operating_variables['PS2']['nominal_value'] * (operating_variables['PS2']['plusminus']/100)
    max_stress = max_stress * 10
    if False:
        print("failure prob. = {}".format(f))
        print("Maximum allowable stress: {}".format(max_stress))
    
    # Get the normalized stress per time step and add the baseline failure probability
    # Cumulative sum of the two to get evolving failure probability accounting for mechanical stress
    #hist000['stress'] = ( np.absolute( hist000['PS1'].diff(periods=-1) ) / max_stress ) + (f * time_step)
    hist000['stress'] = ( np.absolute( hist000.groupby(pid)['PS1'].diff(periods=-1) ) / max_stress ) + (f * time_step)
    hist000['failure_prob'] = hist000['stress'].cumsum()
    hist000 = hist000.drop(columns=['stress'])
    
    eta = -0.05
    gamma = 2
    
    hist000['value'] = hist000['avg_value'] * np.power(hist000['failure_prob'], eta)
    hist000['utility'] = 1 - np.exp(-hist000['value']*gamma)
   
    # Added by Yeojin
    hist000['utility'] = hist000.utility.ffill() # for the last event with NaN failure prob.
    hist000['reward'] = (hist000.groupby(pid).shift(-1).utility - hist000.utility).fillna(0).values
    
    hist000['reward'] = (hist000.reward  - rewardMin)/(rewardMax - rewardMin)

    df.loc[df.TA21s1_org >= 685, 'reward'] -= hazard_mean  # Penalty for the hazard event 

    if False:
        print("\tutility: {}".format(hist000.utility.describe().apply(lambda x: format(x, 'f'))))
        print("\treward: {}".format(hist000.reward.describe().apply(lambda x: format(x, 'f'))))

    return hist000


def getUtility(df, uEnv):
    pid = 'Episode'

    for i, row in uEnv.value_function_limits.iterrows():
        x_min, x_max = row.low_limit, row.up_limit
        xVals = np.arange(x_min, x_max, 0.1)
        nominal = row.nominal
        std = row.nominal - row.low_bound

        yVals = norm.pdf(xVals,nominal,std)/max(norm.pdf(xVals,nominal,std))
        
        df[i+'_value'] = df[i].values
        df[i+'_value'] = df[i+'_value'].apply(lambda x: yVals[ np.argmin(np.abs(xVals-x)) ])
        df.loc[(df[i] >= x_max)|(df[i] <= x_min) ,i+'_value'] = 0
    
    
    if uEnv.key_valNorm == []:
        uEnv.key_valNorm = df[df.time >=  uEnv.normStartTime  ][uEnv.operating_keys].mean().values
    #print("key_valNorm: {}".format(uEnv.key_valNorm))
    # -----------------------------  
    # Get the utility values
    # First get the average sum of the operating variable normal distribution values
    # baseline faiure prob = external value. one of the reactor operating document. 
    
    #print("\n**Before Norm:\n {}".format(df[uEnv.operating_keys].describe()))
    df[uEnv.operating_keys] = df[uEnv.operating_keys].values / uEnv.key_valNorm
    #print("\n**After Norm:\n {}".format(df[uEnv.operating_keys].describe()))
          
    df['avg_value'] = np.mean((df[uEnv.operating_keys].values * uEnv.key_weight), axis=1)
    # -----------------------------
      
    #Get the time step - normalize failure probability per unit s
    time_step = df.groupby(pid).shift(-1)['time'] - df['time']  # Revised by Yeojin

    # Maximum allowable stress is 10x the difference between nominal value and recommanded operating range (rpm)
    # !! This is an assumption !!
    max_stress = uEnv.operating_variables[uEnv.PS2feat]['nominal_value'] * (uEnv.operating_variables[uEnv.PS2feat]['plusminus']/100)
    max_stress = max_stress * 10
    if False:
        print("failure prob. = {}".format(f))
        print("Maximum allowable stress: {}".format(max_stress))
    
    # Get the normalized stress per time step and add the baseline failure probability
    # Cumulative sum of the two to get evolving failure probability accounting for mechanical stress
    #df['stress'] = ( np.absolute( df['PS1'].diff(periods=-1) ) / max_stress ) + (f * time_step)
    df['stress'] = ( np.absolute( df.groupby(pid)['PS1'].diff(periods=-1) ) / max_stress ) +\
                            (uEnv.failProb * time_step)
    df['failure_prob'] = df['stress'].cumsum()
    df = df.drop(columns=['stress'])
    
    df['value'] = df['avg_value'] * np.power(df['failure_prob'], uEnv.eta)
    df['utility'] = 1 - np.exp(-df['value']* uEnv.utilGamma)
   
    # Added 
    df['utility'] = df.utility.ffill() # for the last event with NaN failure prob.
    df['reward'] = ((df.groupby(pid).shift(-1).utility - df.utility)).fillna(0)

    if uEnv.rewardMin == 0:
        uEnv.rewardMin = df.reward.min()
    if uEnv.rewardMax == 0:
        uEnv.rewardMax = df.reward.max()
    df['reward'] = (df.reward  - uEnv.rewardMin)/(uEnv.rewardMax - uEnv.rewardMin) - 0.5 # range [-0.5, 0.5]
    
    # Hazard entering penalty
    df.loc[(df.groupby('Episode').shift(-1).TA21s1_org >= 685)&(df.TA21s1_org<685), 'reward'] = uEnv.hazardCost
    df.loc[(df.groupby('Episode').shift(-1).TA21s1_org >= 685)&(df.TA21s1_org<685), 'utility'] = uEnv.hazardCost
    # Hazard staying penalty
    df.loc[df.TA21s1_org>685, 'reward'] -= uEnv.hazardStayCost
    df.loc[df.TA21s1_org>685, 'utility'] -= uEnv.hazardStayCost
    
    if False:
        print("\tutility: {}".format(df.utility.describe().apply(lambda x: format(x, 'f'))))
        print("\treward: {}".format(df.reward.describe().apply(lambda x: format(x, 'f'))))

    #print(df[uEnv.operating_keys+['avg_value','value','utility']].describe())
    
    return df, uEnv
    
    
def setSimpleReward(df, testMode):
    pid = 'Episode'
    if testMode:
        tgFeat = 'TA21s1'
    else:
        tgFeat = 'TA21s1_org'
    lastEvents = df.groupby(pid).tail(1)
    df['simpleReward'] = 0
    
    df.loc[lastEvents[lastEvents[tgFeat] < 685].index, 'simpleReward'] = 10
    posEp = df[df[tgFeat] >= 685][pid].unique().tolist()
    totEp = df[pid].unique().tolist()
    negEp = [e for e in totEp if e not in posEp]
    negLastEvents = df[df[pid].isin(negEp)].groupby(pid).tail(1)
    df.loc[negLastEvents.index, 'simpleReward'] += 10
    return df



def calReward(uEnv, pdf, feat, trainMean, trainStd, tgTimeList):
    feat_org = [f+'_org' for f in feat]
    pid = 'Episode'
    sdf = pdf[(pdf.time>=uEnv.startTime) & (pdf.time<=uEnv.endTime)].copy(deep=True)
    dropIdx = []
    testvids = sdf[pid].unique().tolist()
    
    # extract every 2 sec between [startTime, endTime] trajectory    
    evalTimeNum = int((uEnv.endTime - uEnv.startTime ) / 2)
    evalTimes = [uEnv.startTime+2*i for i in range(1, evalTimeNum+1)] # 52 ~ 220 sec
    
#     keepIdx = sdf[sdf.time.isin(evalTimes)].index # true eval time events    
    sdf['time'] = (sdf.time).round(0) # round "0.5 sec unit"
    sdf.loc[sdf.time%2==1, 'time'] +=1 # convert odd second to even second
    
#     sdf = pd.concat([sdf.loc[keepIdx], sdf.loc[~sdf.index.isin(keepIdx)]], axis=0)#to give prirority to the true eval time events 
    sdf = sdf.drop_duplicates(['Episode', 'time'])
    sdf = sdf[sdf.time.isin(evalTimes)]
    sdf.reset_index(drop=True, inplace=True) 

    # calcuate reward
    sdf[feat]=np.array(np.array(sdf[feat].values.tolist())*trainStd+trainMean)
#     print("calReward: converted sdf[3feat]:\n{}", sdf[['TA21s1', 'cv42C', 'PS2']].round(2).values[:3, :])
    sdf, uEnv = getUtility(sdf, uEnv)
    sdf = setSimpleReward(sdf, testMode=True)

    avgRwd =  sdf.groupby(pid).reward.sum().mean()
    simRwd = sdf.groupby(pid).simpleReward.sum().mean()
    avgUtil = sdf.groupby(pid).utility.sum().mean()
    
    ## REVISED 112111 
    for i in range(len(testvids)):
        vdf = sdf[sdf[pid]==testvids[i]]
        dropIdx += vdf[vdf.time < tgTimeList[i][0]].index.tolist()
    simuldf = sdf.copy(deep=True) 
    simuldf = simuldf.drop(dropIdx, axis=0) 
    avgSimulUnitUtil = simuldf.groupby(pid).utility.mean().mean()

    if False:
#         print("startTime {}, endTime {} - evalTimes: {}".format(startTime, endTime, evalTimes))
#         print("!!! Eval len(sdf):{}".format(len(sdf)))
        #print("sdf.utility: {}".format(sdf.utility.round(3).tolist()))
        for t in simuldf.time.unique().tolist():
            print("{}({})".format(np.round(t,0), len(simuldf[simuldf.time==t])), end=' ')
        print("")
        
    print("util({:.2f}/{}), unitUtil({:.3f}/{})".format(avgUtil, len(sdf), avgSimulUnitUtil, len(simuldf)), end='\t')
    return sdf, avgRwd, simRwd, avgUtil, avgSimulUnitUtil


def calPosReward(traindf, df, traindf_org):
    df[feat] = np.array(np.array(df[feat].values.tolist())*np.array(traindf_org[feat].std())+np.array(traindf_org[feat].mean()))
    df = getReward0923(df)  
    df = setOverThresholdPenalty(df)
    rwdMax = traindf.reward.max()
    rwdMin = traindf.reward.min()
    #print("reward max: {}, min: {}".format(rwdMax, rwdMin))
    df['reward'] = (np.array(df.reward)-rwdMin)/(rwdMax-rwdMin)
    return df


def calPosReward_test(df, traindf_org, rwdMax, rwdMin):
    df[feat] = np.array(np.array(df[feat].values.tolist())*np.array(traindf_org[feat].std())+np.array(traindf_org[feat].mean()))
    df = getReward0923(df)  
    print(df.reward.describe())
    df = setOverThresholdPenalty(df)
    print(df.reward.describe())
    print("reward max: {}, min: {}".format(rwdMax, rwdMin))
    df['reward'] = (np.array(df.reward)-rwdMin)/(rwdMax-rwdMin)
    print(df.reward.describe())
    return df

## Revised : 101519
# Add penalty 
# when TA21s1 goes over 685: -1
# when TA21s1 goes below 685: +0.5
# when TA21s1 stay over 
def setOverThresholdPenalty(df, penalty):
    df.loc[(df.TA21s1 >= 685) & (df.groupby(pid).shift(1).TA21s1 < 685), 'reward'] += penalty
    return df

def setUnsafeStayPenalty(df):
    df.loc[(df.TA21s1 >= 685), 'reward'] += -.1
    df.loc[(df.TA21s1 >= 680), 'reward'] += -0.1
    return df

#----------------------------------------------------------------------------------------
def initData_env(file, env):
    df = pd.read_csv(file, header=0) 
       
    # set 'done_flag'
    df['done']=0
    df.loc[df.groupby(env.pid).tail(1).index, 'done'] = 1
    # next actions
    df['next_action'] = 0 
    df.loc[:, 'next_action'] = df.groupby(env.pid).Action.shift(-1).fillna(0)
    df['next_action'] = pd.to_numeric(df['next_action'], downcast='integer')
    # df.loc[:, 'next_actions'] = np.array(sdf.groupby('VisitIdentifier').Action.shift(-1).fillna(0), dtype=np.int)

    # next states
    env.nextStateFeat = ['next_'+s for s in env.stateFeat]
    df[env.nextStateFeat] = df.groupby(env.pid).shift(-1).fillna(0)[env.stateFeat]
    
    # action Qs
    Qfeat = ['Q'+str(i) for i in range(len(env.actions))]
    for f in Qfeat:
        df[f] = np.nan
        
    # Temporal Difference for the decay discount factor gamma * exp (-t/tau)
    df.loc[:, 'TD'] = (df.groupby(env.pid)[env.timefeat].shift(-1) - df[env.timefeat]).fillna(0).tolist()
    return df, env


# Set the mean of original key features for vidualization & discrete action features
def initData(traindf, testdf):
    sEnv = SimEnvironment()
    
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


#Make paths for our model and results to be saved in.
def createResultPaths(save_dir, date):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    if not os.path.exists(save_dir+"results"):
        os.mkdir(save_dir+"results")
    if not os.path.exists('results'):
        os.mkdir("results")
    if not os.path.exists('results/'+date):
        os.mkdir('results/'+date)
    print(save_dir)

def setRewardType(rewardType, df, valdf, testdf):
    if rewardType == 'IR':
        print("*** Use IR ")
        IRpath = '../inferredReward/results/'
        irTrain = pd.read_csv(IRpath+'train_IR.csv', header=None)
        irTest = pd.read_csv(IRpath+'test_IR.csv', header=None)
        df['reward'] = irTrain
        valdf['reward'] = irTest
        testdf['reward'] = irTest
    else:
        print("*** Use Delayed Rewards")
    return df, valdf, testdf
 
    

#------------------
# Training

# function is needed to update parameters between main and target network
# tf_vars are the trainable variables to update, and tau is the rate at which to update
# returns tf ops corresponding to the updates

#  Q-network uses Leaky ReLU activation
class Qnetwork():
    def __init__(self, available_actions, state_features, hidden_size, func_approx, myScope):
        self.phase = tf.placeholder(tf.bool)
        self.num_actions = len(available_actions)
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.state = tf.placeholder(tf.float32, shape=[None, len(state_features)],name="input_state")
    
        #if func_approx == 'FC':
        # 4 fully-connected layers ---------------------
        self.fc_1 = tf.contrib.layers.fully_connected(self.state, hidden_size, activation_fn=None)
        self.fc_1_bn = tf.contrib.layers.batch_norm(self.fc_1, center=True, scale=True, is_training=self.phase)
        self.fc_1_ac = tf.maximum(self.fc_1_bn, self.fc_1_bn*0.01)
        self.fc_2 = tf.contrib.layers.fully_connected(self.fc_1_ac, hidden_size, activation_fn=None)
        self.fc_2_bn = tf.contrib.layers.batch_norm(self.fc_2, center=True, scale=True, is_training=self.phase)
        self.fc_2_ac = tf.maximum(self.fc_2_bn, self.fc_2_bn*0.01)
        self.fc_3 = tf.contrib.layers.fully_connected(self.fc_2_ac, hidden_size, activation_fn=None)
        self.fc_3_bn = tf.contrib.layers.batch_norm(self.fc_3, center=True, scale=True, is_training=self.phase)
        self.fc_3_ac = tf.maximum(self.fc_3_bn, self.fc_3_bn*0.01)
        self.fc_4 = tf.contrib.layers.fully_connected(self.fc_3_ac, hidden_size, activation_fn=None)
        self.fc_4_bn = tf.contrib.layers.batch_norm(self.fc_4, center=True, scale=True, is_training=self.phase)
        self.fc_4_ac = tf.maximum(self.fc_4_bn, self.fc_4_bn*0.01)

        # advantage and value streams
        # self.streamA, self.streamV = tf.split(self.fc_3_ac, 2, axis=1)
        self.streamA, self.streamV = tf.split(self.fc_4_ac, 2, axis=1)
                    
        self.AW = tf.Variable(tf.random_normal([hidden_size//2,self.num_actions]))
        self.VW = tf.Variable(tf.random_normal([hidden_size//2,1]))    
        self.Advantage = tf.matmul(self.streamA,self.AW)    
        self.Value = tf.matmul(self.streamV,self.VW)
        #Then combine them together to get our final Q-values.
        self.q_output = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
       
        self.predict = tf.argmax(self.q_output,1, name='predict') # vector of length batch size
        
        #Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.num_actions,dtype=tf.float32)
        
        # Importance sampling weights for PER, used in network update         
        self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)
        
        # select the Q values for the actions that would be selected         
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot), reduction_indices=1) # batch size x 1 vector
        
        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions      
        self.reg_vector = tf.maximum(tf.abs(self.Q)-REWARD_THRESHOLD,0)
        self.reg_term = tf.reduce_sum(self.reg_vector)
        self.abs_error = tf.abs(self.targetQ - self.Q)
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)
        
        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if per_flag:
            self.loss = tf.reduce_mean(self.per_error) + reg_lambda*self.reg_term
        else:
            self.loss = self.old_loss + reg_lambda*self.reg_term

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
        # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
            
        
        
def initialize_model(env, sess, save_dir, df, save_path, init):
    if env.load_model == True:
        print('Trying to load model...')
        try:
            restorer = tf.train.import_meta_graph(save_path + '.meta')
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
            print ("Model restored")
        except IOError:
            print ("No previous model found, running default init")
            sess.run(init)
        try:
            per_weights = pickle.load(open( save_dir + "per_weights.p", "rb" ))
            imp_weights = pickle.load(open( save_dir + "imp_weights.p", "rb" ))

            # the PER weights, governing probability of sampling, and importance sampling
            # weights for use in the gradient descent updates
            df['prob'] = per_weights
            df['imp_weight'] = imp_weights
            print ("PER and Importance weights restored")
            #env.learnRate = 
        except IOError:
            print("No PER weights found - default being used for PER and importance sampling")
    else:
        #print("Running default init")
        sess.run(init)
    #print("Model initialization - done")
    return df#, env
 
 
# -----------------
# Evaluation
   
# extract chunks of length size from the relevant dataframe, and yield these to the caller
# Note: 
# for evaluation, some portion of val/test set can be evaluated, but  
# For test, all the test set's data (40497 events) should be evaluated. Not just 1000 events from the first visit.

def do_eval(sess, env, mainQN, targetQN, df):
    
    gen = process_eval_batch(env, df, df) 
    all_q_ret = []
    phys_q_ret = []
    actions_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    ecr = []
    error_ret = [] #0
    start_traj = 1
    for b in gen: # b: every event for the whole test set
        states,actions,rewards,next_states, _, done_flags, tGammas, _ = b
        # firstly get the chosen actions at the next timestep
        actions_from_q1 = sess.run(mainQN.predict,feed_dict={mainQN.state:next_states, mainQN.phase:0, mainQN.batch_size:len(states)})
        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = sess.run(targetQN.q_output,feed_dict={targetQN.state:next_states, targetQN.phase:0, targetQN.batch_size:len(next_states)})
        # handles the case when a trajectory is finished
        end_multiplier = 1 - done_flags
        # target Q value using Q values from target, and actions from main
        double_q_value = Q2[range(len(Q2)), actions_from_q1]
        # definition of target Q
        if ('Expo' in env.character) or ('Hyper' in env.character):
            targetQ = rewards + (tGammas * double_q_value * end_multiplier)            
        else:
            targetQ = rewards + (env.gamma * double_q_value * end_multiplier)

        # get the output q's, actions, and loss
        q_output, actions_taken, abs_error = sess.run([mainQN.q_output,mainQN.predict, mainQN.abs_error], \
            feed_dict={mainQN.state:states, mainQN.targetQ:targetQ, mainQN.actions:env.actions,
                       mainQN.phase:False, mainQN.batch_size:len(states)})

        # return the relevant q values and actions
        phys_q = q_output[range(len(q_output)), actions]
        agent_q = q_output[range(len(q_output)), actions_taken]
        
#       update the return vals
        error_ret.extend(abs_error)
        all_q_ret.extend(q_output)
        phys_q_ret.extend(phys_q)
        actions_ret.extend(actions)        
        agent_q_ret.extend(agent_q)
        actions_taken_ret.extend(actions_taken)
        ecr.append(agent_q[0])
  
    return all_q_ret, phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, error_ret, ecr


def process_eval_batch(env, df, data):
    a = data.copy(deep=True)    
    actions = np.squeeze(a.Action.tolist())
    next_actions = np.squeeze(a.next_action.tolist()) 
    rewards = np.squeeze(a[env.rewardFeat].tolist())
    done_flags = np.squeeze(a.done.tolist())
    
    if env.maxSeqLen > 1: # LSTM
        states = makeX_event_given_batch(df, a, env.stateFeat, env.pid, env.maxSeqLen)
        next_states = makeX_event_given_batch(df, a, env.nextStateFeat,  env.pid, env.maxSeqLen)   
    else:
        states = a.loc[:, env.stateFeat].values.tolist() 
        next_states =  a.loc[:, env.nextStateFeat].values.tolist()
    
    tGammas = np.squeeze(a.loc[:, env.discountFeat].tolist())

    yield (states, actions, rewards, next_states, next_actions, done_flags, tGammas, a)

        
def do_eval_pdqn_split(sess, env, mainQN, targetQN, gen):
    all_q_ret = []
    phys_q_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    error_ret = []
    
    #for b in gen: # gen: a set of every event (b) with same pred_res (not visit) 
    #    print("b", np.shape(b))
    states, actions, rewards, next_states, _, done_flags, tGammas, selected = gen
    # firstly get the chosen actions at the next timestep
    actions_from_q1 = sess.run(mainQN.predict,feed_dict={mainQN.state:next_states, mainQN.phase:0,\
                                                         mainQN.batch_size:len(states)})
    # Q values for the next timestep from target network, as part of the Double DQN update
    Q2 = sess.run(targetQN.q_output,feed_dict={targetQN.state:next_states, targetQN.phase:0, targetQN.batch_size:len(states)})
    # handles the case when a trajectory is finished
    end_multiplier = 1 - done_flags
    # target Q value using Q values from target, and actions from main
    double_q_value = Q2[range(len(Q2)), actions_from_q1]

    # definition of target Q
    if ('Expo' in env.character) or ('Hyper' in env.character):
        targetQ = rewards + (tGammas * double_q_value * end_multiplier)            
    else:
        targetQ = rewards + (env.gamma * double_q_value * end_multiplier)

    # get the output q's, actions, and loss
    q_output, actions_taken, abs_error = sess.run([mainQN.q_output,mainQN.predict, mainQN.abs_error], \
                                          feed_dict={mainQN.state:states,
                                          mainQN.targetQ:targetQ, 
                                          mainQN.actions:actions,
                                          mainQN.phase:False,
                                          mainQN.batch_size:len(states)})

    # return the relevant q values and actions
    phys_q = q_output[range(len(q_output)), actions]
    agent_q = q_output[range(len(q_output)), actions_taken]

    # update the return vals
    error_ret.extend(abs_error)
    all_q_ret.extend(q_output)
    phys_q_ret.extend(phys_q) 
    agent_q_ret.extend(agent_q)
    actions_taken_ret.extend(actions_taken)

    return all_q_ret, phys_q_ret, agent_q_ret, actions_taken_ret, error_ret, selected


    
def do_eval_pdqn_lstm(sess, env, mainQN, targetQN, testdf, testAll): 
    if env.DEBUG:
        print("do_eval_pdqn_lstm")
    np.set_printoptions(precision=2)
    
    all_q,phys_q,agent_q,actions_taken,error,selected = do_eval_pdqn_split(sess, env, mainQN, targetQN, testAll) 

    testdf.loc[selected.index, 'target_action'] = actions_taken
    testdf.loc[selected.index, 'target_q'] = agent_q
    testdf.loc[selected.index, 'phys_q'] = phys_q
    testdf.loc[selected.index, 'error'] = error
    testdf.loc[selected.index, env.Qfeat] = np.array(all_q)  # save all_q to dataframe      
    
    ecr_ret = testdf.groupby(env.pid).head(1).target_q
        
    return testdf, ecr_ret



def do_save_results(sess, mainQN, targetQN, df, val_df, test_df, state_features, next_states_feat, gamma, save_dir):
    # get the chosen actions for the train, val, and test set when training is complete.
    _, _, _, agent_q_train, agent_actions_train, _, ecr = do_eval(sess, env, mainQN, targetQN, df)
    #print ("Saving results - length IS ", len(agent_actions_train))
    with open(save_dir + 'dqn_normal_actions_train.p', 'wb') as f:
        pickle.dump(agent_actions_train, f)
    _, _, _, agent_q_test, agent_actions_test, _, ecr = do_eval(sess, env, mainQN, targetQN, test_df)   
    
    # save everything for later - they're used in policy evaluation and when generating plots
    with open(save_dir + 'dqn_normal_actions_train.p', 'wb') as f:
        pickle.dump(agent_actions_train, f)

    with open(save_dir + 'dqn_normal_actions_test.p', 'wb') as f:
        pickle.dump(agent_actions_test, f)
        
    with open(save_dir + 'dqn_normal_q_train.p', 'wb') as f:
        pickle.dump(agent_q_train, f)

    with open(save_dir + 'dqn_normal_q_test.p', 'wb') as f:
        pickle.dump(agent_q_test, f)
        
    with open(save_dir + 'ecr_test.p', 'wb') as f:
        pickle.dump(ecr, f)    
    return



def check_convergence(df, agent_actions):
    df["agent_actions"] = agent_actions
    Diff_policy = len(df[df.agent_actions != df.agent_actions_old])
    if Diff_policy > 0:
        print("Policy is not converged {}/{}".format(Diff_policy, len(df)))
    elif Diff_policy == 0:
        print("Policy is converged!!")
    df['agent_actions_old'] = df.agent_actions
    return df

#------------------
# Preprocessing

def process_train_batch(df, size, per_flag, state_features, next_states_feat):
    if per_flag:
        # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)

    actions = a.loc[:, 'Action'].tolist()
    rewards = a.loc[:, 'reward'].tolist()
    states = a.loc[:, state_features].values.tolist()
    
    # next_actions = a.groupby('VisitIdentifier').Action.shift(-1).fillna(0).tolist()
    next_states = a.loc[:, next_states_feat].values.tolist() #a.groupby('VisitIdentifier')[state_features].shift(-1).fillna(0).values.tolist()
    done_flags = a.done.tolist()
    
    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)


# ---------------------------------------------------------------------------- 
# Prediction

# make data with prediction index (1-hour aggregation) 
# def makeXY_idx(pred_df, feat, pid, label, MRL): 
#     X = []
#     Y = []
#     posvids = pred_df[pred_df[label] == 1][pid].unique()
#     eids = pred_df[pid].unique()
#     for eid in eids:
#         edf = pred_df[pred_df[pid] == eid]
#         tmp = np.array(edf[feat])       
#         indexes = edf[edf.pred_idx ==1].index
#         if eid in posvids:
#             Y += [1]*len(indexes)
#         else:
#             Y += [0]*len(indexes)
# 
#         for i in indexes:
#             X.append(pad_sequences([tmp[:i+1]], maxlen = MRL, dtype='float'))
# 
#     return X, Y
# 
# 
# # B. Event-level sequence data generation
# # predict the next label: shift the labels with 1 timestep backward
# def makeXY_event_label(df, feat, pid, label, MRL): 
#     X = []
#     Y = []
#     posvids = df[df[label] == 1][pid].unique()
#     eids = df[pid].unique()
#     for eid in eids:
#         edf = df[df[pid] == eid]
#         tmp = np.array(edf[feat])
#         
#         for i in range(len(tmp)):
#             X.append(pad_sequences([tmp[:i+1]], maxlen = MRL, dtype='float')) 
#             
#         if eid in posvids: # generate event-level Y labels based on the ground truth
#             Y += [1]*len(edf)
#         else:
#             Y += [0]*len(edf)
#     print("df:{} - Xpad:{}, Ypad{}".format(len(df), np.shape(X), np.shape(Y)))
#     return np.array(X), np.array(Y)
# 
# # B. Event-level sequence data generation
# # predict the next label: shift the labels with 1 timestep backward
# def makeXY_event_label2(df, feat, pid, label, MRL): 
#     X = []
#     Y = []
#     posvids = df[df[label] == 1][pid].unique()
#     eids = df[pid].unique()
#     for eid in eids:
#         edf = df[df[pid] == eid]
#         tmp = np.array(edf[feat])
# #X.append(np.array(df.loc[df[pid] == vid, feat])) 
#         for i in range(len(edf)):
#             X.append(np.array(tmp[:i+1]))#, maxlen = MRL, dtype='float')) 
#             
#         if eid in posvids: # generate event-level Y labels based on the ground truth
#             Y += [1]*len(edf)
#         else:
#             Y += [0]*len(edf)
#    
#     X = pad_sequences(X, maxlen = MRL, dtype='float')
#     #print("df:{} - Xpad:{}, Ypad{}".format(len(df), np.shape(X), np.shape(Y)))
#     return X, Y
# 
# 
# def makeX_event_given_batch(df, a, feat, pid, MRL): 
#     X = []
#     eids = a[pid].tolist()
#     idx = a.index.tolist()
#     for i in range(len(eids)):
#         edf = df[df[pid] == eids[i]]
#         tmp = np.array(edf[feat])  
#         X.append(np.array(tmp[:idx[i]]))            
#     X = pad_sequences(X, maxlen = MRL, dtype='float')
#     return X
# 
# 
# def prediction(X, Y, batch_indexes, model): #, bf_mode=True, mi_mode=mi_mode, totfeat=totfeat, fill_mode=fill_mode):
#     predY = []
#     trueY = []       
#     pred = []
#     avg_loss = 0
#     avg_acc = 0
#     trueY = np.array([Y[i] for i in batch_indexes])
#     for j in batch_indexes:
#         loss, acc = model.test_on_batch(np.expand_dims(X[j][0], axis=0), np.array([Y[j]]))
#         pred_val = model.predict_on_batch(np.expand_dims(X[j][0], axis=0))[0][0]
#         predY.append(int(round(pred_val))) # binary prediction
#         pred.append(pred_val) # real value of prediction
#         avg_loss += loss
#         avg_acc += acc
#     model.reset_states()
#     avg_loss /= len(batch_indexes)
#     avg_acc /= len(batch_indexes)
#     #K.clear_session()
#     return pred, predY, trueY, avg_loss, avg_acc
# 
