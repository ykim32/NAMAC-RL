import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import pickle
import copy
import random
# import tbmlib as tl
# import lib_preproc as lp
from keras.preprocessing.sequence import pad_sequences
import os 
from keras.models import model_from_json
import sys
import argparse
import TQN as tq
import datetime
import time
#------------------------------------------------------------------------------------------------------
# Simulator

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


from keras.models import model_from_json
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
    discountFeat = 'DynamicDiscount' #'DecayDiscount'
    rewardFeat = 'reward'#'Reward'

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
        

        
class SimEnvironment():
    simulatorName = ''
    simMaxSeqLen = 5
    polMaxSeqLen = 5
    feat = ['FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C', 'cv43C'] 
    feat_org = [f+'_org' for f in feat]
    actFeat = ['a_' + str(i) for i in range(26)]
    
    ps2_mean = 0
    ps2_std = 0
    ct_mean = 0
    ct_std = 0
    cv_mean = 0
    cv_std = 0
    
    outputNum = 0
    n_features = 0
    polFeat = feat
    simFeat = ['TD'] + feat[:]
    
    policy_sess = ''
    mainQN = ''
    targetQN = ''

    policy_sess2 = ''
    mainQN2 = ''
    targetQN2 = ''

    mPolicy_sess = []
    mMainQN = []
    mTargetQN = []
    
    mpolicy_sess3 = ''
    mainQN3 = ''
    targetQN3 = ''

    mpolicy_sess4 = ''
    mainQN4 = ''
    targetQN4 = ''

    mpolicy_sess5 = ''
    mainQN5 = ''
    targetQN5 = ''
    
#     def __init__(self, feat, actFeat):
        


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
        
def setGPU(tf, env):
        # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto()
    env.config.gpu_options.per_process_gpu_memory_fraction = 0.05
    return env

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
    #print("polEnv.stateFeat:", polEnv.stateFeat)
    mainQN, targetQN, saver, init = setNetwork(tf, polEnv)
    policy_sess = tf.Session(config=polEnv.config) 
    load_RLmodel(policy_sess, tf, policyName)
    
    # get RL test policies for evaluations
    #testdf = pd.read_csv("../data/preproc_pdqn/TBM_zero/3_3_TDQN_belief2_test_0518.csv", header=0) #setRLmodel('../data/preproc_pdqn/TMB_zero/one_lstm_LSTM_FC2_b1_g93_h128_res10_120s/')
   # polTestdf = setRLmodel(policy_dir, policyEpoch=policyEpoch)
    
    return polEnv, policy_sess, mainQN, targetQN #, polTestdf

def setNetwork(tf, env):
    tf.reset_default_graph()

    if env.keyword == 'lstm':
        mainQN = tq.RQnetwork(env, 'main')
        targetQN = tq.RQnetwork(env, 'target')
    elif env.keyword == 'lstm_s2':
        mainQN = tq.RQnetwork2(env, 'main')
        targetQN = tq.RQnetwork2(env, 'target')

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    return mainQN, targetQN, saver, init    

def load_RLmodel(sess, tf, save_dir):
    startTime = time.time()
    try: # load RL model
        restorer = tf.train.import_meta_graph(save_dir + 'ckpt.meta')
        restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
        
#         print ("Model restoring time: {:.2f} sec".format((time.time()-startTime)))
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

        
def setValueFunction(dataset_keys, operating_variables):
    col_list = ['low_limit', 'low_bound', 'nominal', 'up_bound', 'up_limit']
    value_function_limits = pd.DataFrame(columns=col_list)

    for col in dataset_keys[1:]:
        if operating_variables[col]['operating_var']:

            low_bound = operating_variables[col]['nominal_value'] * (1 - (operating_variables[col]['plusminus']/100) )
            low_limit = operating_variables[col]['nominal_value'] * (1 - 3*(operating_variables[col]['plusminus']/100) )
            up_bound = operating_variables[col]['nominal_value'] * (1 + (operating_variables[col]['plusminus']/100) )
            up_limit = operating_variables[col]['nominal_value'] * (1 + 3*(operating_variables[col]['plusminus']/100) )
            nominal = operating_variables[col]['nominal_value']

            data = pd.DataFrame({ 'low_limit': [low_limit], 'low_bound': [low_bound], 'nominal': [nominal],\
                                 'up_bound': [up_bound], 'up_limit': [up_limit] }, index=[col])

            value_function_limits = value_function_limits.append(data, sort=False)
    return value_function_limits


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
#         if 'utilOrg' in dataType:
#             value = []
#             for j in range(0, len(hist000['time'])): 
#                 x_target = hist000[i][j]
#                 x_diff = np.abs(xVals - x_target)
#                 if x_target < x_max and x_target > x_min:    
#                     value.append( yVals [ np.argmin(x_diff) ] )
#                 else:
#                     value.append(0)
#             hist000[i+'_value'] = value
#         else:
        hist000[i+'_value'] = hist000[i].values
        hist000[i+'_value'] = hist000[i+'_value'].apply(lambda x: yVals[ np.argmin(np.abs(xVals-x)) ])
        hist000.loc[(hist000[i] >= x_max)|(hist000[i] <= x_min) ,i+'_value'] = 0
    
    # Get the utility values
    # First get the average sum of the operating variable normal distribution values
    # baseline faiure prob = external value. one of the reactor operating document. 
#     if weightFlag:
#         hist000['avg_value'] = np.mean((hist000[operating_keys].values / key_valNorm * key_weight), axis=1)
#         print("key_valNorm: {}, key_weight: {}".format(key_valNorm, key_weight))
#         print("hist000[avg_value]:{}".format(hist000['avg_value'].round(2))
#     else:
    hist000['avg_value'] = hist000[operating_keys].mean(axis=1)

    # Approximate an evolving failure probability based on mechanical stress
    # The baseline probability of failure (failures per s) for the pump is
    # Based on EBR II Operating docs
    f = 3./3./365./24./60./60.
    
    #Get the time step - normalize failure probability per unit s
    #time_step = history000['time'][1] - history000['time'][0]
    #time_step = hist000['time'][1] - hist000['time'][0]  # Revised by Yeojin
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
    if False:
        print("\tutility: {}".format(hist000.utility.describe().apply(lambda x: format(x, 'f'))))
        print("\treward: {}".format(hist000.reward.describe().apply(lambda x: format(x, 'f'))))

    return hist000

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


# 혹시 컨버지 도중과 컨버지 된 유틸리티를 나눠서 계산해 봐야 하나? 레귤과 비교할 때. 일랩 시뮬레이터가 불안정하면 점수가 막 깍일 수도..
def calReward(pdf, startTime, endTime, feat, trainMean, trainStd, value_function_limits, operating_variables, dataType,\
              tgTimeList, key_valNorm, key_weight):
    feat_org = [f+'_org' for f in feat]
    pid = 'Episode'
    sdf = pdf[(pdf.time>=startTime) & (pdf.time<=endTime)].copy(deep=True)
    dropIdx = []
    testvids = sdf[pid].unique().tolist()
    
    # extract every 2 sec between [startTime, endTime] trajectory    
    evalTimeNum = int((endTime - startTime ) / 2)
    evalTimes = [startTime+2*i for i in range(1, evalTimeNum+1)] # 52 ~ 220 sec
    
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

    sdf = getReward0923(sdf, value_function_limits, operating_variables, dataType, key_valNorm, key_weight)  
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
#     df.loc[(df.TA21s1 >= 690) & (df.groupby(pid).shift(1).TA21s1 < 690), 'reward'] += -2
    df.loc[(df.TA21s1 >= 685) & (df.groupby(pid).shift(1).TA21s1 < 685), 'reward'] += penalty
    #df.loc[(df.TA21s1 >= 680) & (df.groupby(pid).shift(1).TA21s1 < 680), 'reward'] += -.5
    #df.loc[(df.TA21s1 < 685) & (df.groupby(pid).shift(1).TA21s1 >= 685), 'reward'] += .5
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


# def initData(file, pid, stateFeat, timefeat, actions):
#     df = pd.read_csv(file, header=0) 
#     # set 'done_flag'
#     df['done']=0
#     df.loc[df.groupby(pid).tail(1).index, 'done'] = 1
#     # next actions
#     df['next_action'] = 0 
#     df.loc[:, 'next_action'] = df.groupby(pid).Action.shift(-1).fillna(0)
#     df['next_action'] = pd.to_numeric(df['next_action'], downcast='integer')
#     # df.loc[:, 'next_actions'] = np.array(sdf.groupby('VisitIdentifier').Action.shift(-1).fillna(0), dtype=np.int)

#     # next states
#     nextStateFeat = ['next_'+s for s in stateFeat]
#     df[nextStateFeat] = df.groupby(pid).shift(-1).fillna(0)[stateFeat]
    
#     # action Qs
#     Qfeat = ['Q'+str(i) for i in range(len(actions))]
#     for f in Qfeat:
#         df[f] = np.nan
        
#     # Temporal Difference for the decay discount factor gamma * exp (-t/tau)
#     df.loc[:, 'TD'] = (df.groupby(pid)[timefeat].shift(-1) - df[timefeat]).fillna(0).tolist()
#     return df, nextStateFeat

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
 
    
def setAgeFlag(df, test_df, age, hdf):
    splitter = 'AgeFlag'
    trainAge_std = 17.452457113348597
    trainAge_mean = 63.8031197301855 
    df[splitter] = 0
    df.loc[df[df['Age'] * trainAge_std + trainAge_mean >= age].index, splitter] = 1
    test_df[splitter] = 0
    test_df.loc[test_df[test_df['Age']*trainAge_std +trainAge_mean >= age].index, splitter] = 1

    train1 = len(df[df[splitter]==1].VisitIdentifier.unique())
    train0 = len(df[df[splitter]==0].VisitIdentifier.unique())
    test1 = len(test_df[test_df[splitter]==1].VisitIdentifier.unique())
    test0 = len(test_df[test_df[splitter]==0].VisitIdentifier.unique())
    info = "AgeFlag:{} Train - 1({}) 0({}) / Test - 1({}) 0({})".format(age, train1, train0, test1, test0)
    print(info)
    hdf.loc[len(hdf)] = ['splitInfo', info]
    #adf_train = pd.read_csv("../data/preproc/3_3_beforeShock_Prediction_Train_0123.csv", header=0)
    #adf_test = pd.read_csv("../data/preproc/3_3_beforeShock_Prediction_Test_0123.csv", header=0)
    #adf_train.groupby(pid).Age.mean().mean()
    #adf_train.groupby(pid).Age.mean().std()
    
    return df, test_df, splitter, hdf


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
            
            
def update_target_graph(tf_vars,tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:int(total_vars/2)]):
        op_holder.append(tf_vars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tf_vars[idx+int(total_vars/2)].value())))
    return op_holder

def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def update_targetupdate_t (op_holder,sess):
    for op in op_holder:
        sess.run(op)
        
        
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
    states,actions,rewards,next_states, _, done_flags, tGammas, selected = gen
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
                                          feed_dict={mainQN.state:states,mainQN.targetQ:targetQ, mainQN.actions:actions,
                                          mainQN.phase:False, mainQN.batch_size:len(states)})

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


# def do_eval_pdqn(sess, env, data, mainQN, targetQN, testdf):
    
#     if env.streamNum >= 2:
#         for i in range(env.streamNum):
#             gen = process_eval_batch(env, testdf, data.splitTest[i]) 
#             all_q,phys_q,agent_q,actions_taken,error,selected = do_eval_pdqn_split(sess, env, mainQN, targetQN, gen, i) 
#             #print("do_eval_pdqn:idx {}, actions_taken {}, agent_q {}".format(len(idx),len(actions_taken_ret),len(agent_q_ret)))
#             testdf.loc[selected.index, 'target_action'] = actions_taken
#             testdf.loc[selected.index, 'target_q'] = agent_q
#             testdf.loc[selected.index, 'phys_q'] = phys_q
#             testdf.loc[selected.index, 'error'] = error
#             testdf.loc[selected.index, env.Qfeat] = np.array(all_q)  # save all_q to dataframe      

#     elif env.streamNum == 1: # No split (e.g. pred_val as a feature)
#         all_q, phys_q, actions, agent_q, actions_taken, error, ecr = do_eval(sess, mainQN, targetQN,splitTest[0], 
#                   stateFeats, nextStateFeats, gamma, character, numfeat, timesteps = 1)
#         testdf.loc[:, 'target_action'] = actions_taken
#         testdf.loc[:, 'target_q'] = agent_q
#         testdf.loc[:, 'phys_q'] = phys_q
#         testdf.loc[:, 'error'] = error
#         testdf.loc[:, env.Qfeat] = np.array(all_q)   # save all_q to dataframe     
#     else:
#         print("No case for the current splitting (see do_eval_pdqn)")
#         return
    
#     ecr_ret = testdf.groupby(env.pid).head(1).target_q
        
#     return testdf, ecr_ret
      
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

def do_eval_sarsa(sess, env, mainQN, targetQN, df, traindf):
    all_q_ret = []
    phys_q_ret = []
    actions_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    error_ret = 0
    ecr = []

    gen = process_eval_batch(env, df, df) # use the whole data
    for b in gen: # b: the trajectory for one visit
        states, actions, rewards, next_states, next_actions, done_flags, _ = b
        # firstly get the chosen actions at the next timestep
        actions_from_q1 = sess.run(mainQN.predict,feed_dict={mainQN.state:next_states, mainQN.phase : 0})
        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = sess.run(targetQN.q_output,feed_dict={targetQN.state:next_states, targetQN.phase : 0})
        # handles the case when a trajectory is finished
        end_multiplier = 1 - done_flags
        # target Q value using Q values from target, and actions from main
        next_state_q = Q2[range(len(Q2)), next_actions]
        # definition of target Q
        targetQ = rewards + (env.gamma * next_state_q * end_multiplier)

        # get the output q's, actions, and loss
        q_output, actions_taken, abs_error = sess.run([mainQN.q_output, mainQN.predict, mainQN.abs_error], \
            feed_dict={mainQN.state:states, mainQN.targetQ:targetQ, mainQN.actions:actions, mainQN.phase:False})

        # return the relevant q values and actions
        phys_q = q_output[range(len(q_output)), actions]    
        agent_q = q_output[range(len(q_output)), actions_taken]
        
#       update the return vals
        error = np.mean(abs_error)
        all_q_ret.extend(q_output)
        phys_q_ret.extend(phys_q)
        agent_q_ret.extend(agent_q)
        actions_taken_ret.extend(actions_taken)   
        
    df.loc[:, 'target_action'] = actions_taken_ret
    df.loc[:, 'target_q'] = agent_q_ret
    df.loc[:, 'phys_q'] = phys_q_ret
    df.loc[:, 'error'] = error_ret
    df.loc[:, Qfeat] = np.array(all_q_ret)  # save all_q to dataframe      
    ecr_ret = df.groupby(env.pid).head(1).target_q
    return df, ecr_ret
    

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
#     with open(save_dir + 'dqn_normal_actions_val.p', 'wb') as f:
#         pickle.dump(agent_actions_val, f)
    with open(save_dir + 'dqn_normal_actions_test.p', 'wb') as f:
        pickle.dump(agent_actions_test, f)
        
    with open(save_dir + 'dqn_normal_q_train.p', 'wb') as f:
        pickle.dump(agent_q_train, f)
#     with open(save_dir + 'dqn_normal_q_val.p', 'wb') as f:
#         pickle.dump(agent_q_val, f)
    with open(save_dir + 'dqn_normal_q_test.p', 'wb') as f:
        pickle.dump(agent_q_test, f)
        
    with open(save_dir + 'ecr_test.p', 'wb') as f:
        pickle.dump(ecr, f)    
    return


def do_save_results_sarsa(sess, env, mainQN, targetQN, df, val_df, test_df, save_dir):
    # get the chosen actions for the train, val, and test set when training is complete.
    all_q_ret, phys_q_train, phys_actions_train, _, ecr =  do_eval_sarsa(sess, env, mainQN, targetQN, df)        
    all_q_ret, phys_q_test, phys_actions_test, _, ecr = do_eval_sarsa(sess, env, mainQN, targetQN, test_df)
    
    # save everything for later - they're used in policy evaluation and when generating plots
    with open(save_dir + 'phys_actions_train.p', 'wb') as f:
        pickle.dump(phys_actions_train, f)
    with open(save_dir + 'phys_actions_test.p', 'wb') as f:
        pickle.dump(phys_actions_test, f)
        
    with open(save_dir + 'phys_q_train.p', 'wb') as f:
        pickle.dump(phys_q_train, f)
    with open(save_dir + 'phys_q_test.p', 'wb') as f:
        pickle.dump(phys_q_test, f)
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

def convert_action(df, col):
    df.loc[df[df[col] == 0].index, col] = 'N'  #0
    df.loc[df[df[col] == 1].index, col] = 'V'  #1
    df.loc[df[df[col] == 2].index, col] = 'A'  #10
    df.loc[df[df[col] == 3].index, col] = 'AV' #11
    df.loc[df[df[col] == 4].index, col] = 'O'  #100
    df.loc[df[df[col] == 5].index, col] = 'OV' #101
    df.loc[df[df[col] == 6].index, col] = 'OA' #110
    df.loc[df[df[col] == 7].index, col] = 'OAV'#111
    return df
    
def action_dist(df, feat):
    for a in action8:
        print("{}\t{}\t({:.2f})".format(a, len(df[df[feat] == a]), len(df[df[feat] == a])/len(df)))

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


def process_train_batch_sarsa(df, size, per_flag, state_features, next_states_feat):
    if per_flag:   # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)

    actions = a.loc[:, 'Action'].tolist()
    rewards = a.loc[:, 'reward'].tolist()
    states = a.loc[:, state_features].values.tolist()
    next_actions = a.loc[:, 'next_action'].tolist() #np.array(a.groupby('VisitIdentifier').Action.shift(-1).fillna(0), dtype=np.int).tolist()
    next_states =  a.loc[:, next_states_feat].values.tolist() #a.groupby('VisitIdentifier')[state_features].shift(-1).fillna(0).values.tolist()
    done_flags = a.done.tolist()
    
    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(next_actions), np.squeeze(done_flags), a)    
# ---------------
# Analysis
def compareSimilarity(df):
    pid = 'VisitIdentifier' 
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    col = 'Sim_Action'
    df[col] = 0
    df.loc[df[df.Phys_Action == df.Target_Action].index, col] = 1
    df.loc[df[((df.Phys_Action=='V')&(df.Target_Action=='AV'))|((df.Phys_Action=='V')&(df.Target_Action=='OV'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='AV'))|((df.Phys_Action=='A')&(df.Target_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='O')&(df.Target_Action=='OV'))|((df.Phys_Action=='O')&(df.Target_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='OAV'))|((df.Phys_Action=='O')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='V')&(df.Target_Action=='OAV'))].index,col] = 1/3
    df.loc[df[((df.Target_Action=='V')&(df.Phys_Action=='AV'))|((df.Target_Action=='V')&(df.Phys_Action=='OV'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='AV'))|((df.Target_Action=='A')&(df.Phys_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='O')&(df.Phys_Action=='OV'))|((df.Target_Action=='O')&(df.Phys_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='OAV'))|((df.Target_Action=='O')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='V')&(df.Phys_Action=='OAV'))].index,col] = 1/3
    df.loc[df[((df.Target_Action=='OA')&(df.Phys_Action=='OAV'))|((df.Target_Action=='OV')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='AV')&(df.Phys_Action=='OAV'))].index,col] = 2/3
    df.loc[df[((df.Phys_Action=='OA')&(df.Target_Action=='OAV'))|((df.Phys_Action=='OV')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='AV')&(df.Target_Action=='OAV'))].index,col] = 2/3
    
    # Result Analysis 
    rdf = (df.groupby(pid)[col].sum() / df.groupby(pid).size()).reset_index(name='Sim_ratio').sort_values(['Sim_ratio'], ascending=False)
    rdf['Shock'] = 0
    posvids = df[df.Shock == 1].VisitIdentifier.unique()
    rdf.loc[rdf[pid].isin(posvids), 'Shock']  = 1
    
    return df, rdf

def compareDefference(df):
    pid = 'VisitIdentifier' 
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df['Diff_Action'] = 0
    df.loc[df[df.Phys_Action != df.Target_Action].index, 'Diff_Action'] = 1
    df.loc[df[((df.Phys_Action=='V')&(df.Target_Action=='AV'))|((df.Phys_Action=='V')&(df.Target_Action=='OV'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='AV'))|((df.Phys_Action=='A')&(df.Target_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Phys_Action=='O')&(df.Target_Action=='OV'))|((df.Phys_Action=='O')&(df.Target_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='OAV'))|((df.Phys_Action=='O')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='V')&(df.Target_Action=='OAV'))].index,'Diff_Action'] = 0.6
    df.loc[df[((df.Target_Action=='V')&(df.Phys_Action=='AV'))|((df.Target_Action=='V')&(df.Phys_Action=='OV'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='AV'))|((df.Target_Action=='A')&(df.Phys_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Target_Action=='O')&(df.Phys_Action=='OV'))|((df.Target_Action=='O')&(df.Phys_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='OAV'))|((df.Target_Action=='O')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='V')&(df.Phys_Action=='OAV'))].index,'Diff_Action'] = 0.6
    df.loc[df[((df.Target_Action=='OA')&(df.Phys_Action=='OAV'))|((df.Target_Action=='OV')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='AV')&(df.Phys_Action=='OAV'))].index,col] = 0.3  
    df.loc[df[((df.Phys_Action=='OA')&(df.Target_Action=='OAV'))|((df.Phys_Action=='OV')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='AV')&(df.Target_Action=='OAV'))].index,col] = 0.3
    #df['Diff_Action_Inc'] = 0
    #compute the septic shock ratio according to the difference

    # Result Analysis 
    rdf = (df.groupby(pid).Diff_Action.sum() / df.groupby(pid).size()).reset_index(name='Diff_ratio').sort_values(['Diff_ratio'], ascending=False)
    rdf['Shock'] = 0
    posvids = df[df.Shock == 1].VisitIdentifier.unique()
    rdf.loc[rdf[pid].isin(posvids), 'Shock']  = 1
    # Diff_ratio by range: shock probability (but underlying distribution?)
    # [0~0.1), [0.1, 0.2), ... [0.9, 1], [1] 
    #
    return df, rdf

def getPolicySimilarity(df, div):
    df.rename(columns = {'Agent_Action': 'Target_Action'}, inplace=True)    
    _, resdf = compareSimilarity(df)
    resdf, shockSimRate, nonShockSimRate, avgSimRate = getPolicySimRate(resdf, div)
    return resdf, shockSimRate, nonShockSimRate, avgSimRate

def getPolicySimRate(rdf, div): # div: how many 
    col = 'Sim_ratio'
    res = pd.DataFrame(columns=['SimRate', 'pos', 'neg', 'shockRate'])
    avgSimRate = rdf[col].mean()
    shockSimRate = rdf[rdf.Shock==1][col].mean()
    nonShockSimRate = rdf[rdf.Shock==0][col].mean()
    print("Similarity Rate: avg({:.4f}), shock({:.4f}), non-shock({:.4f})".format(avgSimRate, shockSimRate, nonShockSimRate))
    #print("DiffRate PosNum\tNegNum\tShockRate")   
    for i in range(div):
        s0 = len(rdf[(rdf[col] < (i+1)/div) & (rdf[col] >= i/div) & (rdf.Shock == 0)]) 
        s1 = len(rdf[(rdf[col] < (i+1)/div) & (rdf[col] >= i/div) & (rdf.Shock == 1)])
        if s0+s1 > 0:
            #print("{:0.2f}\t {:} \t{:} \t{:0.2f}".format((i)/div, s1, s0, s1/(s0+s1)))
            res.loc[len(res),:] = [(i)/div, s1, s0, s1/(s0+s1)]
    s0 = len(rdf[(rdf[col] == 1) & (rdf.Shock == 0)]) 
    s1 = len(rdf[(rdf[col] == 1) & (rdf.Shock == 1)])
    if s0+s1 > 0:
        #print("1 \t {:} \t{} \t{:.2f}".format(s1, s0, s1/(s0+s1)))
        res.loc[len(res),:] = [1, s1, s0, s1/(s0+s1)]
    return res, shockSimRate, nonShockSimRate, avgSimRate

def getPolicyDiffRate(rdf, div):
    resDiff = pd.DataFrame(columns=['diffRate', 'pos', 'neg', 'shockRate'])
    avgDiffRate = rdf.Diff_ratio.mean()
    shockDiffRate = rdf[rdf.Shock==1].Diff_ratio.mean()
    nonShockDiffRate = rdf[rdf.Shock==0].Diff_ratio.mean()
    print("Difference Rate: avg({:.4f}), shock({:.4f}), non-shock({:.4f})".format(avgDiffRate, shockDiffRate, nonShockDiffRate))
    #print("DiffRate PosNum\tNegNum\tShockRate")
    for i in range(div):
        s0 = len(rdf[(rdf.Diff_ratio < (i+1)/div) & (rdf.Diff_ratio >= i/div) & (rdf.Shock == 0)]) 
        s1 = len(rdf[(rdf.Diff_ratio < (i+1)/div) & (rdf.Diff_ratio >= i/div) & (rdf.Shock == 1)])
        if s0+s1 > 0:
            #print("{:0.2f}\t {:} \t{:} \t{:0.2f}".format((i)/div, s1, s0, s1/(s0+s1)))
            resDiff.loc[len(resDiff),:] = [(i)/div, s1, s0, s1/(s0+s1)]

    s0 = len(rdf[(rdf.Diff_ratio == 1) & (rdf.Shock == 0)]) 
    s1 = len(rdf[(rdf.Diff_ratio == 1) & (rdf.Shock == 1)])
    if s0+s1 > 0:
        #print("1 \t {:} \t{} \t{:.2f}".format(s1, s0, s1/(s0+s1)))
        resDiff.loc[len(resDiff),:] = [1, s1, s0, s1/(s0+s1)]
    return resDiff, avgDiffRate, shockDiffRate, nonShockDiffRate
        
def showTrajectoryLength(df):
    groupSize = df.groupby('VisitIdentifier').size()
    print("Length of Trajectory: mean({:.2}), max({}), min({})".format(groupSize.mean(), groupSize.max(), groupSize.min()))
        
def rl_analysis(df, target_actions, all_q_ret, target_q, availableActions):
    pid = 'VisitIdentifier' 
    df.loc[:, 'target_action'] = target_actions
    df.loc[:, 'target_q'] = target_q

    # save all_q to dataframe
    for i in range(np.size(all_q_ret, 1)):
        df['Q'+str(i)] = np.array(all_q_ret)[:,i]

    df['Target_Action'] = np.nan
    df['Phys_Action'] = np.nan
    idx = 0
    for a in availableActions: 
        df.loc[df[df.target_action == idx].index, 'Target_Action'] = a
        df.loc[df[df.Action == idx].index, 'Phys_Action'] = a
        idx += 1    

    action_num = len(df.Action.unique())
    df.loc[:, 'rand_action'] = [random.randint(0, action_num-1) for _ in range(len(df))]
    
    # TODO: For each trajectories, calculate the difference between the actions from the physicians and the agent
    df['Diff_Action'] = 0
    df.loc[df[df.Action != df.target_action].index, 'Diff_Action'] = 1
    #df['Diff_Action_Inc'] = 0
    #compute the septic shock ratio according to the difference

    # Result Analysis 
    rdf = (df.groupby(pid).Diff_Action.sum() / df.groupby(pid).size()).reset_index(name='Diff_ratio').sort_values(['Diff_ratio'], ascending=False)
    rdf['Shock'] = 0
    posvids = df[df.Shock == 1].VisitIdentifier.unique()
    rdf.loc[rdf[pid].isin(posvids), 'Shock']  = 1
    # Diff_ratio by range: shock probability (but underlying distribution?)
    # [0~0.1), [0.1, 0.2), ... [0.9, 1], [1] 
    #
    return df, rdf

def rl_analysis_pdqn(env, df): 
    df['Target_Action'] = np.nan
    df['Phys_Action'] = np.nan
    idx = 0
    for a in env.actions: 
        df.loc[df[df.target_action == idx].index, 'Target_Action'] = a
        df.loc[df[df.Action == idx].index, 'Phys_Action'] = a
        idx += 1    

    action_num = len(df.Action.unique())
    df.loc[:, 'rand_action'] = [random.randint(0, action_num-1) for _ in range(len(df))]
    
    # TODO: For each trajectories, calculate the difference between the actions from the physicians and the agent
    df['Diff_Action'] = 0
    df.loc[df[df.Action != df.target_action].index, 'Diff_Action'] = 1
    #df['Diff_Action_Inc'] = 0
    #compute the septic shock ratio according to the difference

    # Result Analysis 
    rdf = (df.groupby(env.pid).Diff_Action.sum() / df.groupby(env.pid).size()).reset_index(name='Diff_ratio').sort_values(['Diff_ratio'], ascending=False)
    rdf[env.label] = 0
    posvids = df[df[env.label] == 1][env.pid].unique()
    rdf.loc[rdf[env.pid].isin(posvids), 'Shock']  = 1
    # Diff_ratio by range: shock probability (but underlying distribution?)
    # [0~0.1), [0.1, 0.2), ... [0.9, 1], [1] 

    return df, rdf

def rewardNonShock(df, negvids, reward):
    ndf = df[df[pid].isin(negvids)]
    df.loc[ndf.groupby(pid).tail(1).index, 'reward'] = reward # non-shock: positive reward
    return df

def extractResults(feat, phy, phy_ir, dqn, dqn_ir):
    rdf = pd.DataFrame(columns = ['timestep', 'Physician', 'Physician_IR', 'DQN', 'DQN_IR'])
    rdf.loc[:, 'timestep'] = phy.timestep
    rdf.loc[:, 'Physician'] = phy[feat]
    rdf.loc[:, 'Physician_IR'] = phy_ir[feat]
    rdf.loc[:, 'DQN'] = dqn[feat]
    rdf.loc[:, 'DQN_IR'] = dqn_ir[feat]
    return rdf

def mergeResults(path, keyword):
    phy = pd.read_csv(path+"log_"+keyword+"_phy_SR.csv", header=0)
    phy_ir = pd.read_csv(path+"log_"+keyword+"_phy_IR.csv", header=0)
    dqn = pd.read_csv(path+"log_"+keyword+"_dqn_SR.csv", header=0)
    dqn_ir = pd.read_csv(path+"log_"+keyword+"_dqn_IR.csv", header=0)

    avg_q = extractResults('avg_Q', phy, phy_ir, dqn, dqn_ir)
    mae = extractResults('MAE', phy, phy_ir, dqn, dqn_ir)
    avg_loss = extractResults('avg_loss', phy, phy_ir, dqn, dqn_ir)

    avg_q.to_csv(path+keyword+"_avg_Q.csv", index=False)
    mae.to_csv(path+keyword+"_mae.csv", index=False)
    avg_loss.to_csv(path+keyword+"_avg_loss.csv", index=False)

# ---------------------------------------------------------------------------- 
# Prediction

def prep_predData(file, feat, taus, pid):
    df = pd.read_csv(file, header=0)
    df = df[['VisitIdentifier', 'MinutesFromArrival']+feat]
    df = df.drop(df.loc[df.MinutesFromArrival < 0].index)  # Drop negative MinutesFromArrival events
    # get the idx to make the prediction data (with 1-hour time window)
    df['agg_idx'] = np.abs(df.MinutesFromArrival // 60) 
    df['pred_idx'] = 0
    df.loc[(df.shift(-1).agg_idx - df.agg_idx) != 0, 'pred_idx'] = 1 
    
    df = tl.setmi(df, feat) # set missing indicators
    df = tl.make_belief(df, pid, feat, taus, mode='.75') # impute 
    return df

# TBM imputation + Add MI
def init_predict(pid):
    filepath = '../../rl/data/preproc/'    
    taus_org = pd.read_csv('data/pdqn_final_LSTM/timedf0h.csv', header = 0)
    taus = taus_org.loc[1, :][1:]
    feat = taus_org.columns[1:].tolist()
    totfeat = [] # for prediction    
    for f in feat:
        totfeat.append(f)
        totfeat.append(f+'_mi')
    pred_train_df = prep_predData(filepath+"3_3_beforeShock_Prediction_Train_0123.csv", feat, taus, pid)
    pred_test_df = prep_predData(filepath+"3_3_beforeShock_Prediction_Test_0123.csv", feat, taus, pid)

    return taus, feat, totfeat, pred_train_df, pred_test_df 

# make data with prediction index (1-hour aggregation) 
def makeXY_idx(pred_df, feat, pid, label, MRL): 
    X = []
    Y = []
    posvids = pred_df[pred_df[label] == 1][pid].unique()
    eids = pred_df[pid].unique()
    for eid in eids:
        edf = pred_df[pred_df[pid] == eid]
        tmp = np.array(edf[feat])       
        indexes = edf[edf.pred_idx ==1].index
        if eid in posvids:
            Y += [1]*len(indexes)
        else:
            Y += [0]*len(indexes)

        for i in indexes:
            X.append(pad_sequences([tmp[:i+1]], maxlen = MRL, dtype='float'))

    return X, Y

# get prediction values (Currently used)
from keras import backend as K
def get_prediction(df, test_df, pid, label, MRL):
    MRL = 20
    mi_mode = True
    fill_mode = '.75'    
    # initialize the prediction data
    taus, feat, totfeat, pred_train_df, pred_test_df  = init_predict(pid) # TBM imputation + add MI
    print("RL data: train({}), test({}) / Pred data idx: train({}), test({})".format(len(df),len(test_df),pred_train_df.pred_idx.sum(), pred_test_df.pred_idx.sum()))
    
    # make X, Y data for prediction (RL data and XY data have same indexes)
    test_Xpad, test_Ypad = makeXY_idx(pred_test_df, totfeat, pid, label, MRL)
    train_Xpad, train_Ypad = makeXY_idx(pred_train_df, totfeat, pid, label, MRL)
    
    df['pred_val'] = np.nan
    test_df['pred_val'] = np.nan
    train_pred_val = []
    test_pred_val = []
    with tf.Session() as sess_pred:
        K.set_session(sess_pred)
        pred_model = tl.load_model('data/pdqn_final_LSTM/models/model_final_.75True')
        pred_model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics = ['binary_accuracy'])
        #loss, acc = pred_model.test_on_batch(np.expand_dims(train_Xpad[0][0], axis=0), np.array([Y[0]]))
        for j in df.index:
            train_pred_val.append(pred_model.predict_on_batch(np.expand_dims(train_Xpad[j][0], axis=0))[0][0])
            if j % 10000 == 0:
                print(j)
        df['pred_val'] = train_pred_val
        for j in test_df.index:
            test_pred_val.append(pred_model.predict_on_batch(np.expand_dims(test_Xpad[j][0], axis=0))[0][0])
            if j % 10000 == 0:
                print(j)
        test_df['pred_val'] = test_pred_val

    df.to_csv("../data/preproc/10_BS_pred_train.csv", index=False)
    test_df.to_csv("../data/preproc/10_BS_pred_test.csv", index=False)
    return df, test_df

# B. Event-level sequence data generation
# predict the next label: shift the labels with 1 timestep backward
def makeXY_event_label(df, feat, pid, label, MRL): 
    X = []
    Y = []
    posvids = df[df[label] == 1][pid].unique()
    eids = df[pid].unique()
    for eid in eids:
        edf = df[df[pid] == eid]
        tmp = np.array(edf[feat])
        
        for i in range(len(tmp)):
            X.append(pad_sequences([tmp[:i+1]], maxlen = MRL, dtype='float')) 
            
        if eid in posvids: # generate event-level Y labels based on the ground truth
            Y += [1]*len(edf)
        else:
            Y += [0]*len(edf)
    print("df:{} - Xpad:{}, Ypad{}".format(len(df), np.shape(X), np.shape(Y)))
    return np.array(X), np.array(Y)

# B. Event-level sequence data generation
# predict the next label: shift the labels with 1 timestep backward
def makeXY_event_label2(df, feat, pid, label, MRL): 
    X = []
    Y = []
    posvids = df[df[label] == 1][pid].unique()
    eids = df[pid].unique()
    for eid in eids:
        edf = df[df[pid] == eid]
        tmp = np.array(edf[feat])
#X.append(np.array(df.loc[df[pid] == vid, feat])) 
        for i in range(len(edf)):
            X.append(np.array(tmp[:i+1]))#, maxlen = MRL, dtype='float')) 
            
        if eid in posvids: # generate event-level Y labels based on the ground truth
            Y += [1]*len(edf)
        else:
            Y += [0]*len(edf)
   
    X = pad_sequences(X, maxlen = MRL, dtype='float')
    #print("df:{} - Xpad:{}, Ypad{}".format(len(df), np.shape(X), np.shape(Y)))
    return X, Y


def makeX_event_given_batch(df, a, feat, pid, MRL): 
    X = []
    eids = a[pid].tolist()
    idx = a.index.tolist()
    for i in range(len(eids)):
        edf = df[df[pid] == eids[i]]
        tmp = np.array(edf[feat])  
        X.append(np.array(tmp[:idx[i]]))            
    X = pad_sequences(X, maxlen = MRL, dtype='float')
    return X

def imputeTBM(df, pid, fill_mode, outfile):
    staticFeat = ['Gender', 'Age', 'Race'] 
    taus_org = pd.read_csv('data/pdqn_final_LSTM/timedf0h.csv', header = 0)
    taus = taus_org.loc[1, :][1:]
    feat = taus_org.columns[1:].tolist()#[:14]
    df = tl.make_belief_mean(df, pid, feat, taus, mode='.75')    
    df.to_csv(outfile, index=False)
    
    totfeat = [] # for prediction    
    for f in feat:
        totfeat.append(f)
        totfeat.append(f+'_mi')
    #totfeat += staticFeat
    #feat += staticFeat
    return df, feat, totfeat

def imputeTBM_GAR(df, pid, numFeat, staticFeat, fill_mode, mi_mode, taufile, outfile, outWindow):      
    print("load TBM imputed data with GAR")
    #numFeat = ['HeartRate', 'Temperature', 'SystolicBP', 'MAP', 'Lactate', 'WBC', 'Platelet', 'Creatinine',
    #   'RespiratoryRate', 'FIO2', 'PulseOx', 'BiliRubin', 'BUN',  'Bands']#The order should be fixed for both prediction and RL
    #staticFeat = ['Gender', 'Age', 'Race'] 
    taus_org = pd.read_csv(taufile, header = 0)
    taus = taus_org.loc[1, :][1:]
    tauFeat = taus_org.columns[1:].tolist()#[:14]
    if outWindow == 'zero':
        df = tl.make_belief(df, pid, tauFeat, taus, mode='.75')  #FOor both standardization and normalization case, zero-impute should be done out of the reliable time windows. (Here the meaning is that we assume when we cannot rely on the previous features, we just exclude thouse missing features and let LSTM infer it only based on other features. It'll find the relationalship.      
    elif outWindow == 'mean':
        df = tl.make_belief_mean(df, pid, tauFeat, taus, mode='.75')    
    
    df.to_csv(outfile, index=False)
   
    if mi_mode == True:
        df = tl.setmi(df, numFeat) # set missing indicators
        totfeat = [] # for prediction    
        for f in numFeat:
            totfeat.append(f)
            totfeat.append(f+'_mi')
        totfeat += staticFeat
        feat = numFeat + staticFeat
    else:
        feat = numFeat + staticFeat
        totfeat = feat
    return df, feat, totfeat

def get_prediction_raw(df, pid, label, totfeat, pred_model, MRL, outfile):  
    # make X, Y data for prediction (RL data and XY data have same indexes)
    print("make X, Y with event level labels")
    Xpad, Ypad = makeXY_event_label(df, totfeat, pid, label, MRL)
    df['visitShock'] = Ypad
    df['pred_val'] = np.nan
    pred_val = []
    pred_Y = []
    print("Prediction start...")
    with tf.Session() as sess_pred:
        K.set_session(sess_pred)
        #pred_model = tl.load_model('data/pdqn_GAR_LSTM/models/model_final_.75True')
        pred_model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics = ['binary_accuracy'])
        #loss, acc = pred_model.test_on_batch(np.expand_dims(train_Xpad[0][0], axis=0), np.array([Y[0]]))
        for j in df.index:
            pred_val.append(pred_model.predict_on_batch(np.expand_dims(Xpad[j][0], axis=0))[0][0])
            if j % 5000 == 0 and j > 0:
                print(j)
               
        df['pred_val'] = pred_val
        print("{}: pos - pred {} / ground {} with 0.5".format(j, len(df[df.pred_val>=0.5]), np.sum(Ypad)))

        if outfile != '':
            df.to_csv(outfile, index=False)
    return df


def prediction(X, Y, batch_indexes, model): #, bf_mode=True, mi_mode=mi_mode, totfeat=totfeat, fill_mode=fill_mode):
    predY = []
    trueY = []       
    pred = []
    avg_loss = 0
    avg_acc = 0
    trueY = np.array([Y[i] for i in batch_indexes])
    for j in batch_indexes:
        loss, acc = model.test_on_batch(np.expand_dims(X[j][0], axis=0), np.array([Y[j]]))
        pred_val = model.predict_on_batch(np.expand_dims(X[j][0], axis=0))[0][0]
        predY.append(int(round(pred_val))) # binary prediction
        pred.append(pred_val) # real value of prediction
        avg_loss += loss
        avg_acc += acc
    model.reset_states()
    avg_loss /= len(batch_indexes)
    avg_acc /= len(batch_indexes)
    #K.clear_session()
    return pred, predY, trueY, avg_loss, avg_acc

# Not used
def predict(model, X, df, keyword):
    pred_val = []
    for j in df.index:
        pred_val.append(model.predict_on_batch(np.expand_dims(X[j][0], axis=0))[0][0])
        if j % 10000 == 0:
            print(keyword, j)
    df['pred_val'] = pred_val
    df.to_csv("data/preproc/10_BS_pred_"+keyword+".csv", index=False)
    return df

# Not used
def multiProcPredict(model, train_Xpad, test_Xpad, df, test_df):
    p = mp.Process(target=predict, args = (pred_model, train_Xpad, df, "train"))
    threads.append(p)
    p.start()

    p = mp.Process(target=predict, args = (pred_model, test_Xpad, test_df, "test"))
    threads.append(p)
    p.start()

    for p in threads:
        p.join()
    return df, test_df


    
# ----------------------------------------------------------------------------
# IS  

def get_behavior_prob(df):
    print(" - Action probability")
    behavior_prob = []
    actions = df.action.unique().tolist()
    for i in actions: 
        behavior_prob.append(len(df[df.action == i])/len(df))
    for i in range(len(actions)):
	    print("action {}: {:.4f}".format(i, behavior_prob[i]))
    return behavior_prob
    
      
def compute_IS(test_df, save_dir, behavior_df, theta, gamma):
    columns = ['userID','action','reward','Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7','bQ0','bQ1','bQ2','bQ3','bQ4','bQ5','bQ6','bQ7']
    agent_df = pd.DataFrame(columns = columns)    
    agent_df[['userID','action','reward','Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7']] = test_df[['VisitIdentifier','target_action','reward','Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7']]
    agent_df[['bQ0','bQ1','bQ2','bQ3','bQ4','bQ5','bQ6','bQ7']] = behavior_df[['Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7']]
    agent_df.to_csv(save_dir+"target_results.csv", index=False)
        
    phy_df = pd.DataFrame(columns = columns)
    phy_df[columns] = behavior_df[['VisitIdentifier','target_action','reward','Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7']]  
    phy_df.to_csv(save_dir+"phy_results.csv", index=False)
#     agent_df.columns = columns
#     rand_df.columns = columns

#     agent_df.to_csv(save_dir+"agent_results.csv", index=False)
#     rand_df.to_csv(save_dir+"rand_results.csv", index=False)
    
    print("- Read data: Agent")
    agent_IS = IS(save_dir+"target_results.csv", theta, gamma)
    agent_IS.readData()
    print("- Read data: Physician (Behavior)")
    phys_IS = IS(save_dir+"phy_results.csv", theta, gamma)
    phys_IS.readData()
#     print("- Random")
#     rand_IS = IS(save_dir+"rand_results.csv", theta, gamma)
#     rand_IS.readData()
    print("      Agent     Physician  Random")
    print("IS:   {:.2f}\t{:.2f}\t{:.2f}".format(agent_IS.IS(), phys_IS.IS()))
    print("WIS:  {:.2f}\t{:.2f}\t{:.2f}".format(agent_IS.WIS(), phys_IS.WIS()))#, rand_IS.IS()))
    print("PDIS: {:.2f}\t{:.2f}\t{:.2f}".format(agent_IS.PDIS(), phys_IS.PDIS()))#, rand_IS.IS()))
    return phy_df, agent_df

def showIS(save_dir):
    for theta in [0.01, 0.1, 1.0]:
        agent_IS = IS(save_dir+"agent_results.csv", theta, gamma)
        agent_IS.readData()
        phys_IS = IS(save_dir+"phy_results.csv", theta, gamma)
        phys_IS.readData()
        rand_IS = IS(save_dir+"rand_results.csv", theta, gamma)
        rand_IS.readData()
        print("      Agent     Physician  Random")
        print("IS:   {:.2f}\t{:.2f}    {:.2f}".format(agent_IS.IS(), phys_IS.IS(), rand_IS.IS()))
        print("WIS:  {:.2f}\t{:.2f}    {:.2f}".format(agent_IS.WIS(), phys_IS.WIS(), rand_IS.IS()))
        print("PDIS: {:.2f}\t{:.2f}    {:.2f}".format(agent_IS.PDIS(), phys_IS.PDIS(), rand_IS.IS()))
     
     
def getActionProb(Qs, action, theta):
    # target_prob
    Q_act = Qs[action]
    sum_Q = sum(math.exp(x*theta) for x in Qs)
    if sum_Q == 0:
        prob = 0
    else:
        prob = math.exp(Q_act*theta) / sum_Q
    return prob
    
class IS(object):
    def __init__(self, filename, theta, gamma):
        self.filename = filename
        #self.behavior_file = behavior_file
        self.theta = theta
        self.gamma = gamma
        self.traces = []
        self.n_action = 0
        self.n_user = 0
        self.random_prob = 0
        self.behavior_prob =  [] 

    def readData(self):
        tdf = pd.read_csv(self.filename)
        self.n_action = 8
        Q_index = tdf.columns.get_loc("reward") + 1
        Q_list = list(tdf)[Q_index:Q_index+self.n_action]
        bQ_list = list(tdf)[Q_index+self.n_action:Q_index+2*self.n_action]
        user_list = list(tdf['userID'].unique())

        self.n_user = len(user_list)
        self.random_prob = 1.0 / self.n_action
        # self.behavior_prob = [0.4431, 0.0009, 0.4610, 0.0016, 0.0293, 0.0010, 0.0600, 0.0031]
		#test: [0.4431, 0.0009, 0.4610, 0.0016, 0.0293, 0.0010, 0.0600, 0.0031]
        #training: [0.4719, 0.0018, 0.4465, 0.0016, 0.0292, 0.0017, 0.0447, 0.0027]
        for user in user_list:
            user_sequence = []
            user_data = tdf.loc[tdf['userID'] == user,]
            row_index = user_data.index.tolist()

            for i in range(0, len(row_index)):
                action = user_data.loc[row_index[i], 'action']
                reward = user_data.loc[row_index[i], 'reward']
                Qs = user_data.loc[row_index[i], Q_list].tolist()
                behavior_Qs = user_data.loc[row_index[i], bQ_list].tolist()
                user_sequence.append((action, reward, Qs, behavior_Qs))

            self.traces.append(user_sequence)


    def IS(self):
        IS = 0
        for each_trajectory in self.traces:
            #cumul_policy_prob = 1
            #cumul_random_prob = 1
            #cumul_behavior_prob = 1
            #cumulative_reward = 0             
            IS_reward = 0
            #weight = 1

            for i, (action, reward, Qs, behavior_Qs) in enumerate(each_trajectory):
                prob_target = getActionProb(Qs, action, theta)
                prob_behavior = getActionProb(behavior_Qs, action, theta)
                IS_reward += (prob_target / prob_behavior) * math.pow(self.gamma, i) * reward
                #cumul_policy_prob += math.exp(prob_logP)
                #cumul_random_prob += math.exp(self.random_prob)
                #cumul_behavior_prob += math.exp(self.behavior_prob[action])
                #cumulative_reward += math.pow(self.gamma, i) * reward

            #weight = math.log(cumul_policy_prob) / math.log(cumul_random_prob)
            #weight = math.log(cumul_policy_prob) / math.log(cumul_behavior_prob)                
            #IS_reward = cumulative_reward * weight

            IS += IS_reward

        IS = float(IS) / self.n_user
        return IS
        
    def WIS(self):
        total_weight = 0
        WIS = 0
        for each_trajectory in self.traces:
#             cumul_policy_prob = 1
#             cumul_random_prob = 1
#             cumul_behavior_prob = 1
#             cumulative_reward = 0
            IS_reward = 0
            
            for i, (action, reward, Qs) in enumerate(each_trajectory):
                prob_target = getActionProb(Qs, action, theta)
                prob_behavior = getActionProb(behavior_Qs, action, theta)
                weight = prob_target / prob_behavior
                IS_reward += weight * math.pow(self.gamma, i) * reward
                total_weight += weight
                # cumul_policy_prob += math.exp(prob_logP)
#                 cumul_random_prob += math.exp(self.random_prob)
#                 cumul_behavior_prob += math.exp(self.behavior_prob[action])
#                 cumulative_reward += math.pow(self.gamma, i) * reward
            
            #weight = math.log(cumul_policy_prob) / math.log(cumul_random_prob)
#             weight = math.log(cumul_policy_prob) / math.log(cumul_behavior_prob)                
#             total_weight += weight
#             IS_reward = cumulative_reward * weight

            WIS += IS_reward

        WIS = float(WIS) / total_weight
        return WIS


    def PDIS(self):
        PDIS = 0

        for each_trajectory in self.traces:
            #cumul_policy_prob = 1
            #cumul_random_prob = 1
            #cumul_behavior_prob = 1
            PDIS_each_trajectory = 0

            for i, (action, reward, Qs) in enumerate(each_trajectory):
                prob_target = getActionProb(Qs, action, theta)
                prob_behavior = getActionProb(behavior_Qs, action, theta)
                weight = prob_target / prob_behavior
                IS_reward += weight * math.pow(self.gamma, i) * reward
                    
                #cumul_policy_prob += math.exp(prob_logP)
                #cumul_random_prob += math.exp(self.random_prob)
                #cumul_behavior_prob += math.exp(self.behavior_prob[action])
                #weight = math.log(cumul_policy_prob) / math.log(cumul_random_prob)
                #weight = math.log(cumul_policy_prob) / math.log(cumul_behavior_prob)                
                PDIS_each_trajectory += math.pow(self.gamma, i) * reward * weight

            PDIS += PDIS_each_trajectory

        PDIS = float(PDIS) / self.n_user
        return PDIS



# class IS(object):
#     def __init__(self, filename1, theta, gamma):
#         self.filename = filename1
#         self.theta = theta
#         self.gamma = gamma
#         self.traces = []
#         self.n_action = 0
#         self.n_user = 0
#         self.random_prob = 0
#         self.behavior_prob =  [] 
# 
#     def readData(self):
#         raw_data = pd.read_csv(self.filename)
#         Q_index = raw_data.columns.get_loc("reward") + 1
#         Q_list = list(raw_data)[Q_index:]
#         user_list = list(raw_data['userID'].unique())
#         self.n_action = len(Q_list)
#         self.n_user = len(user_list)
#         self.random_prob = 1.0 / self.n_action
#         # self.behavior_prob = [0.4431, 0.0009, 0.4610, 0.0016, 0.0293, 0.0010, 0.0600, 0.0031]
# 		#test: [0.4431, 0.0009, 0.4610, 0.0016, 0.0293, 0.0010, 0.0600, 0.0031]
#         #training: [0.4719, 0.0018, 0.4465, 0.0016, 0.0292, 0.0017, 0.0447, 0.0027]
#         for user in user_list:
#             user_sequence = []
#             user_data = raw_data.loc[raw_data['userID'] == user,]
#             row_index = user_data.index.tolist()
# 
#             for i in range(0, len(row_index)):
#                 action = user_data.loc[row_index[i], 'action']
#                 reward = user_data.loc[row_index[i], 'reward']
#                 Qs = user_data.loc[row_index[i], Q_list].tolist()
# 
#                 user_sequence.append((action, reward, Qs))
# 
#             self.traces.append(user_sequence)
# 
#     def IS(self):
#         IS = 0
# 
#         for each_student_data in self.traces:
#             cumul_policy_prob = 1
#             cumul_random_prob = 1
#             cumul_behavior_prob = 1
#             cumulative_reward = 0
#              
#             weight = 1
# 
#             for i, (action, reward, Qs) in enumerate(each_student_data):
# 
#                 Q_act = Qs[action]
#                 sum_Q = sum(math.exp(x*self.theta) for x in Qs)
#                 if sum_Q == 0:
#                     prob_logP = 0
#                 else:
#                     prob_logP = math.exp(Q_act*self.theta) / sum_Q
#                 #prob_logP = math.exp(Q_act) / sum(math.exp(x*self.theta) for x in Qs)
# 
#                 cumul_policy_prob += math.exp(prob_logP)
#                 cumul_random_prob += math.exp(self.random_prob)
#                 cumul_behavior_prob += math.exp(self.behavior_prob[action])
#                 cumulative_reward += math.pow(self.gamma, i) * reward
# 
#             #weight = math.log(cumul_policy_prob) / math.log(cumul_random_prob)
#             weight = math.log(cumul_policy_prob) / math.log(cumul_behavior_prob)                
#             IS_reward = cumulative_reward * weight
# 
#             IS += IS_reward
# 
#         IS = float(IS) / self.n_user
#         return IS
# 
#     def WIS(self):
#         total_weight = 0
#         WIS = 0
#         for each_student_data in self.traces:
#             cumul_policy_prob = 1
#             cumul_random_prob = 1
#             cumul_behavior_prob = 1
#             cumulative_reward = 0
# 
#             for i, (action, reward, Qs) in enumerate(each_student_data):
# 
#                 Q_act = Qs[action]
#                 sum_Q = sum(math.exp(x*self.theta) for x in Qs)
#                 if sum_Q == 0:
#                     prob_logP = 0
#                 else:
#                     prob_logP = math.exp(Q_act * self.theta) / sum_Q
#                 #prob_logP = math.exp(Q_act) / sum(math.exp(x*self.theta) for x in Qs)
# 
#                 cumul_policy_prob += math.exp(prob_logP)
#                 cumul_random_prob += math.exp(self.random_prob)
#                 cumul_behavior_prob += math.exp(self.behavior_prob[action])
#                 cumulative_reward += math.pow(self.gamma, i) * reward
# 
#             #weight = math.log(cumul_policy_prob) / math.log(cumul_random_prob)
#             weight = math.log(cumul_policy_prob) / math.log(cumul_behavior_prob)                
#             total_weight += weight
#             IS_reward = cumulative_reward * weight
# 
#             WIS += IS_reward
# 
#         WIS = float(WIS) / total_weight
#         return WIS
# 
# 
#     def PDIS(self):
#         PDIS = 0
# 
#         for each_student_data in self.traces:
#             cumul_policy_prob = 1
#             cumul_random_prob = 1
#             cumul_behavior_prob = 1
#             PDIS_each_student = 0
# 
#             for i, (action, reward, Qs) in enumerate(each_student_data):
# 
#                 Q_act = Qs[action]
#                 #prob_logP = math.exp(Q_act) / sum(math.exp(x*self.theta) for x in Qs)
#                 sum_Q = sum(math.exp(x*self.theta) for x in Qs)
#                 if sum_Q == 0:
#                     prob_logP = 0
#                 else:
#                     prob_logP = math.exp(Q_act*self.theta) / sum_Q
#                     
#                 cumul_policy_prob += math.exp(prob_logP)
#                 cumul_random_prob += math.exp(self.random_prob)
#                 cumul_behavior_prob += math.exp(self.behavior_prob[action])
#                 #weight = math.log(cumul_policy_prob) / math.log(cumul_random_prob)
#                 weight = math.log(cumul_policy_prob) / math.log(cumul_behavior_prob)                
#                 PDIS_each_student += math.pow(self.gamma, i) * reward * weight
# 
#             PDIS += PDIS_each_student
# 
#         PDIS = float(PDIS) / self.n_user
#         return PDIS
# 
#     def IS_new(self):
#         IS = 0
# 
#         for each_student_data in self.traces:
#             cumul_policy_prob = 1
#             cumul_random_prob = 10  # 1
#             cumulative_reward = 0
#             weight = 1
# 
#             for i, (action, reward, Qs) in enumerate(each_student_data):
# 
#                 Q_act = Qs[action]
#                 prob_logP = math.exp(Q_act*self.theta) / sum(math.exp(x*self.theta) for x in Qs)
# 
#                 weight *= self.n_action * prob_logP / self.random_prob
#                 cumulative_reward += math.pow(self.gamma, i) * reward
# 
#             IS_reward = cumulative_reward * weight
# 
#             IS += IS_reward
# 
#         IS = float(IS) / self.n_user
#         return IS
        
        
        
# For Test
# test = IS("/Users/song/Desktop/sample.csv", 1, 0.9)
# test.readData()
# print test.IS()
# print test.WIS()
# print test.PDIS()


# ----------------------------        
# TEMP
# from tensorflow.contrib import rnn
# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# '''
# To classify images using a recurrent neural network, we consider every image
# row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
# handle 28 sequences of 28 steps for every sample.
# '''
# def RNN(x, weights, biases):
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, timesteps, n_input)
#     # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
# 
#     # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
#     x = tf.unstack(x, timesteps, 1)
#     # Define a lstm cell with tensorflow
#     lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#     # Get lstm cell output
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#     # Linear activation, using rnn inner loop last output
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Training Parameters
# learning_rate = 0.001
# training_steps = 10000
# batch_size = 128
# display_step = 200
# 
# # Network Parameters
# num_input = 21 # number of state_features 
# timesteps = 28 # timesteps
# num_hidden = 128 # hidden layer num of features
# num_classes = 8 # total classes (8 actions)
# 
# # tf Graph input
# X = tf.placeholder("float", [None, timesteps, num_input])
# Y = tf.placeholder("float", [None, num_classes])
# 
# # Define weights
# weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
# biases = {'out': tf.Variable(tf.random_normal([num_classes]))}
# 
# logits = RNN(X, weights, biases)
# prediction = tf.nn.softmax(logits)
# 
# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
# 
# # Evaluate model (with test logits, for dropout to be disabled)
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 
# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()
# 
# # Start training
# with tf.Session() as sess:
#     # Run the initializer
#     sess.run(init)
#     for step in range(1, training_steps+1):
#         batch_x, batch_y = mnist.train.next_batch(batch_size)
#         # Reshape data to get 28 seq of 28 elements
#         batch_x = batch_x.reshape((batch_size, timesteps, num_input))
#         # Run optimization op (backprop)
#         sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#         if step % display_step == 0 or step == 1:
#             # Calculate batch loss and accuracy
#             loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
#             print("Step " + str(step) + ", Minibatch Loss= " + \
#                   "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.3f}".format(acc))
#     print("Optimization Finished!")
#     # Calculate accuracy for 128 mnist test images
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
#     test_label = mnist.test.labels[:test_len]
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))