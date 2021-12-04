# ========================================================================================
# Author: Yeojin Kim
# Date: July 22, 2021
# File: Reactor control agent class + UtilityEnvironment
# ========================================================================================
import argparse
import datetime
import numpy as np
import random
import os
import pickle
import pandas as pd
from collections import deque  
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect


        
class ReactorAgent(object):
    config = []
    pid = 'Episode'
    label = 'Unsafe'
    timeFeat = 'time'
    discountFeat = 'DynamicDiscount' 
    rewardFeat = 'reward'
    TDmode = False # use time interval as input feature for time-aware state approximation
    
    policySess = ''
    date = ''
    actionNum = 33 
    actFeat = ['a_'+str(i) for i in range(actionNum)]
    actions = [i for i in range(actionNum)] 
    Qfeat = ['Q'+str(i) for i in actions]
    
    numFeat = ['FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C','cv43C'] 
    feat_org = [f+'_org' for f in numFeat if f != 'TD']            
    nextNumFeat = [f + '_next' for f in numFeat]
    stateFeat = numFeat 
    nextStateFeat = [f + '_next' for f in stateFeat]

    actionMag = [       0 ,  96.131027 , 97.46003 , 98.789024 , 100.11803 , 101.44703 , 102.77602 , 104.10503 , 
                105.43403 , 106.76303 , 108.09203 , 109.42103 , 110.75003 , 112.07903 , 113.40803 , 114.73703 , 
                116.06603 , 117.39503 , 118.72403 , 120.05303 , 121.38203 , 122.71103 , 124.04003 , 125.36903 , 
                126.69804 , 128.02704 , 129.35603 , 130.68503 , 132.01404 , 133.34303 , 134.67204 , 136.00104 , 137.33003 ]
    actionDur = [0]+[50]*32
    
    def __init__(self, args, avgTD, train_file):
        self.DEBUG = False         # True: Turn on Debug mode
        self.EVAL = True
        self.method = args.method  #TQN, TState, TDiscount, RQN
        self.DUELING = False
        self.DOUBLE = False
        self.MTA = args.mta
        
        self.keyword = args.k
        self.character = args.c
        
        if 'lstm' in self.keyword:   
            self.func_approx = 'LSTM'
        else:
            self.func_approx = 'FC'
        
        # ----------------------------------------
        # beta_start: increase 0.002 every 1000 iterations from 0.4 up to 1.0 during 300K iterations
        self.reg_lambda = 5
        self.ERROR_CLIP = True # -1 <=TD_ERROR <= 1 for stable learning
        self.Q_clipping = True # for Q-value clipping 
        self.Q_THRESHOLD = 150 # for Q-value clipping
        self.REWARD_THRESHOLD = 100 # for fold 8  # 100
        self.huber_threshold = 1
        
        self.tau_init = self.tau = float(args.tau)
        self.tau = float(args.tau) # { 0.001, 0.01 } Rate to update target network toward primary network        

        self.weightType = args.w  # weight type of reward function
        if '721' in self.weightType:
            self.rewardWeights = [0.7, 0.2, 0.1]
        elif '631' in self.weightType:
            self.rewardWeights = [0.6, 0.3, 0.1]
        else:
            self.rewardWeights = [0.33, 0.33, 0.33]

        self.avgTD = avgTD 
        self.belief = float(args.b)
        self.targetFuture = float(args.tf)  # concerned time window (seconds)
        self.gamma = self.belief**(self.avgTD/self.targetFuture)
        
        self.hidden_size = int(args.hu)
        self.numSteps = int(args.t)
        self.gpuID = str(args.g)

        #self.date = str(datetime.datetime.now().strftime('%m%d%H')) 
        self.date = "071321_CV_t200_m250K"
        self.repeat = 2
        
        self.cutTime = 200            # too late cut after converged trajectories -> limite the multi-view with fixed memory size
        self.MEMORY_SIZE = int(250000) # too small replay buffer -> limite the multi-view 
        self.memory = deque(maxlen=self.MEMORY_SIZE)  # do not delete the oldest experient in offline learning
        #cur_state, actionTrain[tridx], accTI, accReward, next_state, actionTrain[tridx+ti], doneTrain[tridx]
        self.MEM = pd.DataFrame(columns = ['state', 'action', 'ti', 'reward','next_state','next_action','done','prob','imp_weight'])
        self.ACTMEM = False   # reward >= 10
        self.ACTMEM_SIZE = int(10000)     
        self.actMEM_period = 20 # 5% periodical sparse-action training
        self.actMEM = deque(maxlen=self.ACTMEM_SIZE)
        self.train_start = 100000

        # Prioritized Experience Replay ----------
        self.PER = False
        self.per_alpha = 0.6 # alpha: importance weight for sampling (the higher, the more prioritized sampling) 
        self.per_epsilon = 0.01 # PER hyperparameter
        self.beta_start = 0.4 # beta: importance weight for learning (the higher, the more prioritized learning) 
        self.beta_increment = 0.001 #0.0005
        self.beta_period = 2000

        self.impWeights = deque(maxlen=self.MEMORY_SIZE) # same to the training set size
        self.perProb = deque(maxlen=self.MEMORY_SIZE) # same to the training set size
        # ----------------------------------------
        self.maxSeqLen = int(args.msl)
        self.trainMinTI = args.trminTI
        self.trainMaxTI = args.trmaxTI
        self.testMinTI = args.teminTI
        self.testMaxTI = args.temaxTI
        self.UPPER_TI = args.upperTI #0:disabled(the upper bound of time interval for temporal abstraction)
        self.valGen = args.valGen

        self.normTI = 60 # self.trainMaxTI * self.maxSeqLen * self.avgTD

        if ('TQN' in self.method or 'TState' in self.method):
            self.state_input_size = len(self.numFeat) +1
            self.TStateMode = 1
        else:
            self.state_input_size = len(self.numFeat) 
            self.TStateMode = 0
        if ('TQN' in self.method or 'TDiscount' in self.method):
            self.state_input_size = len(self.numFeat) +1
            self.TDiscountMode = 1
        else:
            self.TDiscountMode = 0
            
        self.LEARNING_RATE = args.lr_an # use it or not
        self.learnRate_init = float(args.lr)
        self.learnRate = self.learnRate_init #0.01~0.0001 # init_value (αk+1 = 0.98αk)
        self.learnRate_min = 0.001 #float(self.learnRate_init/100)
        self.learnRateFactor = args.lrf #  0.99
        self.learnRatePeriod = 1000 # init=0.01 : .1M:0.0036 .2M:0.0013, .4M:0.0003 .5M:<0.0001 
        self.LR_TAU = True

        self.load_model = True 

        
        self.save_results = True
            
        self.batch_size = 32
        if 'DEBUG' in self.keyword:
            self.period_eval = 10
        else:
            self.period_eval = 20000
            self.period_save = 500000

        self.fold = int(args.fold)  
        self.hyperParam = "lr{}_{}_tau{}".format(self.LEARNING_RATE, self.learnRate, self.tau)
        self.filename = "{}_{}_ti{}_{}s{}".format(self.method, self.keyword, 
                         self.trainMinTI, self.trainMaxTI, self.maxSeqLen)
        if self.MTA: 
            self.filename = 'MTA_'+ self.filename
#         self.filename = "{}_{}_{}_b{}_g{}_h{}_{}lr{}_tau{}_ti{}_{}_seq{}".format(self.method, self.keyword, 
#                self.weightType, int(self.belief*10), int(self.gamma*100), 
#                self.hidden_size, self.LEARNING_RATE, self.learnRate, self.tau,self.trainMinTI, self.trainMaxTI, self.maxSeqLen)
                         
        if self.DEBUG:
            self.filename = 'DEBUG_' + self.filename
        self.train_file = train_file
        self.resAll = pd.DataFrame(columns = ['epoch', 'utility']) 
                      
        

    def remember(self, state, action, time_interval, reward, next_state, done):
        #experience = state, action, time_interval, reward, next_state, done
        self.memory.append((experience))
        if self.ACTMEM and action > 0:
            self.actMEM.append((experience))
            

    def update_ti_queue(self, time_interval, t_state_queue):
        if self.TStateMode:   
            t_state_queue[-1][-1] = time_interval/(self.normTI)
        return t_state_queue

    # Temporal Abstraction of sequence from the given working memory
    def makeTempAbstract(self, obs, time_interval, workMEM):

        if self.method == 'TQN' or self.method == 'TState': 
            obs = np.array(obs.tolist()+[time_interval/(self.normTI)])
            
        workMEM.append(obs)        
        
        if len(workMEM) > self.maxSeqLen:
            workMEM = workMEM[-self.maxSeqLen:]
        
        # ----------------------------------------------
        # Convert t_state to RNN data format
        X = np.array(workMEM)
        if len(workMEM) < self.maxSeqLen:
            X = np.pad(X, [(self.maxSeqLen-len(workMEM), 0), (0,0)],
                        mode='constant')        
    

        if 'FC' in self.func_approx: #concatenate 'seqDim' number of recent states 
            X = X.flatten()

        t_state = np.expand_dims(X, axis=0)

        return t_state, workMEM    

    # make tuples (o, a, Delta t, r, o', done)
    def make_tuples(self, df):
        obs = np.array(df[self.numFeat])
        actions = np.array(df.Action.tolist())
        
        if 'next_action' not in df.columns:
            df['next_action'] = df.groupby(self.pid).shift(-1).Action.fillna(0)
        next_actions = np.array(df.next_action.tolist()) 
        
        rewards = np.array(df[self.rewardFeat].tolist())
        
        if 'done' not in df.columns:
            df['done'] = 0
            df.loc[df.groupby(self.pid).tail(1).index, 'done'] = 1
        done_flags = np.array(df.done.tolist())
        time_intv = np.array(df.TD.tolist()) 
        return obs, actions, time_intv, rewards, next_actions, done_flags 

    
 
    # For the validation/test data, generate the state inputs 
    def make_MT_StateData(self, df, save_dir):
        
        zeros = np.array([0]*len(self.numFeat))
        statePool = []
        # Init (seqLen * maxTI) size of sequence memory --------------------------------
        len_seqMEM = self.maxSeqLen * self.testMaxTI
        seqMEM = deque(maxlen=int(len_seqMEM))
        for i in range(len_seqMEM-1):
            seqMEM.append((zeros,0))
        # ------------------------------------------------------------------------------         
        # Make training tuples 
        obsL, actionL, tiL, rewardL, next_actionL, doneL = self.make_tuples(df)
        
        # For each event
        for i in range(0, len(obsL)):
            
            seqMEM.append((obsL[i],tiL[i]))
            
            # Select time steps for TA 
            sampleNum = np.min([self.maxSeqLen -1, len(seqMEM)-1])
            timeSteps = random.sample(range(0, len(seqMEM)-1), sampleNum) # select previous time steps
            timeSteps.append(len(seqMEM) - 1) # put a current time step    
            timeSteps.sort()
            
            # Init the working memory for temporal abstraction
            workMEM = []

            # Select (seqLen-1) number of observations and a current observation to generate a current Obs
            for e in range(len(timeSteps)):  
                if e < len(timeSteps)-1: # for previous observations
                    accTI = 0
                    for a in range(timeSteps[e],timeSteps[e+1]):
                        accTI += seqMEM[a][1]
                                    
                else: # for current observation
                    accTI = seqMEM[-1][1]

                cur_state, workMEM = self.makeTempAbstract(seqMEM[timeSteps[e]][0], accTI, workMEM) # Put a current observation
            
            statePool.append(cur_state[0])
    
            if doneL[i] == 1: # init seqMEM when a new visit starts
                # Init (seqLen * maxTI) size of sequence memory -------------------------
                seqMEM = deque(maxlen=int(len_seqMEM))
                for i in range(len_seqMEM-1):
                    seqMEM.append((zeros,0))
                # -----------------------------------------------------------------------

        # save the data -----------------------------------------------------------------
        print("data: {}".format(np.shape(statePool)))
        if ('reg' in self.keyword) or (self.testMinTI == self.testMaxTI):
            valPath = "{}/{}_{}{}_reg{}".format(save_dir, self.method, self.func_approx, self.maxSeqLen, self.testMaxTI)
        else:
            valPath = "{}/M{}_{}{}_ti{}_{}".format(save_dir, self.method, self.func_approx, self.maxSeqLen, 
                                                         self.testMinTI, self.testMaxTI)
        if self.UPPER_TI:
            valPath += '_UPPER{}'.format(self.UPPER_TI)
            
        #if not os.path.exists(valPath):
        with open(valPath+'_val.p', 'wb') as f:
            pickle.dump(statePool, f)
        print(" ! == Generate validataion data: {} == !".format(valPath))
        return statePool


    def process_eval_batch_TA(self, df, state_ti_list):

        a = df.copy(deep=True)
        idx = a.index.values.tolist()
        actions = np.array(a.Action.tolist())
        next_actions = np.array(a.next_action.tolist()) 
        rewards = np.array(a[self.rewardFeat].tolist())
        done_flags = np.array(a.done.tolist())
    
        states = state_ti_list
        next_states = np.concatenate((state_ti_list[1:], [state_ti_list[-1]])) # at the end of trajectory, ignore the next state
        print("state: {}, next_state: {}".format(np.shape(states),np.shape(next_states)))
         
        if self.TDiscountMode: 
            tGammas = np.array(a.loc[:, self.discountFeat].tolist())
            #print("** Set Exponential Discount: {}".format(np.shape(tGammas)))
        else:
            tGammas = []
        
        return (states, actions, rewards, next_states, next_actions, done_flags, tGammas)   

        
        
    # Make a training batch
    def get_mt_batch(self, memory):        
        
        if self.PER: # weighted sampling
            #minibatch = random.choices(mem, weights=self.perProb, k=self.batch_size)
            minibatch = memory.sample(n=self.batch_size, weights=memory.prob)
        else:
            minibatch = memory.sample(n=self.batch_size, weights=None)
            #minibatch = random.sample(memory, min(len(memory), self.batch_size))
        
        if self.func_approx == 'LSTM':
            #print("*** state_input_size: {}".format(np.size(minibatch[0][0])))
            state = np.zeros((self.batch_size, self.maxSeqLen, self.state_input_size))
            next_state = np.zeros((self.batch_size, self.maxSeqLen, self.state_input_size))
        else: # Dense
            state = np.zeros((self.batch_size, self.state_input_size * self.maxSeqLen))
            next_state = np.zeros((self.batch_size, self.state_input_size* self.maxSeqLen))

        
        # do this before prediction
        # for speedup, this could be done on the tensor level
        idx = minibatch.index.values.tolist()
        
        # MEM: ['state', 'action', 'ti', 'reward','next_state','next_action','done','prob','imp_weight'])
        state =  np.array(minibatch.state.values.tolist())#.reshape([self.batch_size, self.maxSeqLen, -1])
        action = np.array(minibatch.action.tolist())
        ti = np.array(minibatch.ti.tolist())
        reward = np.array(minibatch.reward.tolist())
        next_state = np.array(minibatch.next_state.tolist())#.reshape([self.batch_size, self.maxSeqLen, -1])
        next_action = np.array(minibatch.next_action.tolist())
        done = np.array(minibatch.done.tolist())
        
        #print("state: {}, action: {}, ti: {}".format(np.shape(state), np.shape(action), np.shape(ti)))
        if self.TDiscountMode: # Temporal discount function 
            tGammas = self.belief**(ti/self.targetFuture)
        else:
            tGammas = []    

        return state, action, ti, reward, next_state, next_action, done, tGammas, minibatch
    

    # Get the best action 
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))


def parsingPolicy(parser, avgTD, train_file):
    parser.add_argument("-method", type=str, choices={'DQN','TState','TDiscount','TQN'})   # Method: TQN, Tstate, Tdiscount, RQN
    parser.add_argument("-k", type=str, default = 'PDD_lstm')   # {'lstm', 'dense'} keyword for models & results
                                                            
    parser.add_argument("-mta", type=int, default = 0) # multi-temporal abstraction                                                          
    parser.add_argument("-c",  type=str, default= '')       # characteristics of model
    parser.add_argument("-fold", type=int, default = 0)      # fold 
    parser.add_argument("-debug", type=int, default = 0)    # Debug mode
     
    parser.add_argument("-g", type=str, default = '0')       # GPU ID
    # Network hyperparameters
    parser.add_argument("-msl", type=int, default = 3)       # max sequence length for LSTM
    parser.add_argument("-hu", type=int, default = 128)      # hidden_size
    parser.add_argument("-t",  type=int, default = 500000)   # training iteration
    parser.add_argument("-lr_an",  type=int, default = 1) # learning rate - 1: anneal/ 0:fix
    parser.add_argument("-lr",  type=float, default = 0.005) # learning rate
    parser.add_argument("-lrf",  type=float, default = 0.98) # learning rate    
    parser.add_argument("-tau",  type=float, default = 0.005) # Soft update of Target Q-network 
    
    # Time-aware hyperparameters
    parser.add_argument("-d",  type=float, default = 0.982)   # discount factor gamma
    parser.add_argument("-b",  type=float, default = 0.5)    # belief for TQN
    parser.add_argument("-tf", type=int, default = 200)     # task time window (minutes)
    
    # Multi-temporal view 
    parser.add_argument("-trminTI",  type=int, default = 1)   # training min time interval (TI)
    parser.add_argument("-trmaxTI",  type=int, default = 1)   # training max TI
    parser.add_argument("-teminTI",  type=int, default = 1)   # test min TI
    parser.add_argument("-temaxTI",  type=int, default = 1)   # test max TI
    parser.add_argument("-upperTI",  type=int, default = 0)   # set the upper bound TI for temporal abstraction (0: disabled)
    
    parser.add_argument("-valGen",  type=int, default = 0)   # generate a valdiation data set with given options
    
    # Reward weights
    parser.add_argument("-w", type=str, choices = {'w631', 'w721', 'w1'}, default = 'w631') # target future time
    
    args = parser.parse_args()

    env = ReactorAgent(args, avgTD, train_file)
    if env.method=='TQN' or env.method=='TState':
        #print("* Add time intervals as a state input feature")
        env.stateFeat = env.numFeat[:]+['TD']
        env.TDmode=True
        
    env.nextStateFeat = ['next_'+s for s in env.stateFeat]

    # Able to add the following network learning options with keywords 
    # P: prioritized experience replay (Disable for Temporal Abstraction methods)
    # Doub: Double DQN  (Able for all the methods)
    # Duel: Dueling DQN (Able for all the methods)
    if 'P' in env.filename:
        env.PER = True
    if 'Doub' in env.filename or 'DD' in env.filename:
        env.DOUBLE = True
    if 'Duel' in env.filename or 'DD' in env.filename:
        env.DUELING = True    

    print("======================================================")
    print("{} - MTA: {} DOUB: {} (tau={}), DUEL: {}, PER: {} (cutTime: {})".format(env.method,
          env.MTA, env.DOUBLE, env.tau, env.DUELING, env.PER, env.cutTime))
    print("belief: {}, targetFuture: {}, discount: {}".format(env.belief, env.targetFuture, env.gamma))
    print("Memory: {}, actMEM:{} ({}-{}), train_start: {}".format(env.MEMORY_SIZE, env.ACTMEM, env.ACTMEM_SIZE, env.actMEM_period, env.train_start))
    print("trainTI: [{}, {}], maxSeqLen: {}".format(env.trainMinTI, env.trainMaxTI, env.maxSeqLen))
    print("learningRate: {} (anneal:{} min:{} factor:{} period:{})- LR=TAU:{}".format(env.learnRate, env.LEARNING_RATE, env.learnRate_min,
                                                                     env.learnRateFactor, env.learnRatePeriod, env.LR_TAU))   
    print("======================================================")
        
    return env
    
    
