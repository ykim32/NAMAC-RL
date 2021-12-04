import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import numpy as np
import math
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
import copy
import shutil
import argparse
import multiprocessing as mp
import datetime
import time
import lib_dqn_lstm as ld



from functools import reduce
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Save the hyper-parameters
def saveHyperParameters(env):    
    hdf = pd.DataFrame(columns = ['type', 'value'])
    hdf.loc[len(hdf)] = ['file', env.filename]
    hdf.loc[len(hdf)] = ['gamma', env.gamma]
    hdf.loc[len(hdf)] = ['gamma_rate', env.gamma_rate]
    hdf.loc[len(hdf)] = ['hidden_size', env.hidden_size]
    hdf.loc[len(hdf)] = ['batch_size', env.batch_size]

    hdf.loc[len(hdf)] = ['learning_rate', env.learnRate]
    hdf.loc[len(hdf)] = ['learning_rate_factor', env.learnRateFactor]
    hdf.loc[len(hdf)] = ['learning_rate_period', env.learnRatePeriod]
    hdf.loc[len(hdf)] = ['Q_clipping', env.Q_clipping]
    hdf.loc[len(hdf)] = ['Q_THRESHOLD', env.Q_THRESHOLD]
    hdf.loc[len(hdf)] = ['keyword', env.keyword]
    hdf.loc[len(hdf)] = ['character', env.character]
    hdf.loc[len(hdf)] = ['belief', env.belief]
    
    hdf.loc[len(hdf)] = ['numFeat', env.numFeat]
    hdf.loc[len(hdf)] = ['stateFeat', env.stateFeat]
    hdf.loc[len(hdf)] = ['actions', env.actions]
    hdf.loc[len(hdf)] = ['per_flag', env.per_flag]
    hdf.loc[len(hdf)] = ['per_alpha', env.per_alpha]
    hdf.loc[len(hdf)] = ['per_epsilon', env.per_epsilon]
    hdf.loc[len(hdf)] = ['beta_start', env.beta_start]
    hdf.loc[len(hdf)] = ['reg_lambda', env.reg_lambda]
    hdf.loc[len(hdf)] = ['hidden_size', env.hidden_size]
    hdf.loc[len(hdf)] = ['training_iteration', env.numSteps]
    return hdf

def setGPU(tf, env):
        # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto()
    env.config.gpu_options.per_process_gpu_memory_fraction = 0.1
    return env
    
# init for PER important weights and params
# Might be better with optimistic initial values > max reward 
def initPER(df, env):
    df.loc[:, 'prob'] = abs(df[env.rewardFeat])
    temp = 1.0/df['prob']
    temp[temp == float('Inf')] = 1.0
    df.loc[:, 'imp_weight'] = pow((1.0/len(df) * temp), env.beta_start)
    return df




#  Recurrent Q-network / Deep Q-network
class RQnetwork():
    def __init__(self, env, myScope):
        self.phase = tf.placeholder(tf.bool)
        self.num_actions = len(env.actions)
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        #self.hidden_state = tf.placeholder(tf.float32, shape=[None, env.hidden_size],name="hidden_state")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        if env.func_approx=='LSTM': # 1-layer of LSTM + 2-layer of FC
            self.state = tf.placeholder(tf.float32, shape=[None, env.maxSeqLen, len(env.stateFeat)], name="input_state")
            
            lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(env.hidden_size),rnn.BasicLSTMCell(env.hidden_size)])
            self.state_in = lstm_cell.zero_state(self.batch_size,tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(\
                    inputs=self.state, cell=lstm_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope+'_rnn')
            self.rnn_output = tf.unstack(self.rnn, env.maxSeqLen, 1)[-1]
            #self.streamA, self.streamV = tf.split(self.rnn_output, 2, axis=1)

            self.fc1, self.fc1_bn, self.fc1_ac = setHiddenLayer(self.rnn_output, env.hidden_size, self.phase, last_layer=0)
            self.fc2, self.fc2_bn, self.fc2_ac = setHiddenLayer(self.fc1_ac, env.hidden_size, self.phase, last_layer=1)
            self.fc_out = self.fc2_ac

        elif env.func_approx == 'FC': # 4 fully-connected layers 
            self.state = tf.placeholder(tf.float32, shape=[None, len(env.stateFeat)*env.maxSeqLen], name="input_state")
            
            self.fc1, self.fc1_bn, self.fc1_ac = setHiddenLayer(self.state, env.hidden_size, self.phase, last_layer=0)
            self.fc2, self.fc2_bn, self.fc2_ac = setHiddenLayer(self.fc1_ac, env.hidden_size, self.phase, last_layer=0)
            self.fc3, self.fc3_bn, self.fc3_ac = setHiddenLayer(self.fc2_ac, env.hidden_size, self.phase, last_layer=0)
            self.fc4, self.fc4_bn, self.fc4_ac = setHiddenLayer(self.fc3_ac, env.hidden_size, self.phase, last_layer=1)        
            self.fc_out = self.fc4_ac
            
        # advantage and value streams
        self.streamA, self.streamV = tf.split(self.fc_out, 2, axis=1)
                           
        self.AW = tf.Variable(tf.random_normal([env.hidden_size//2,self.num_actions])) 
        self.VW = tf.Variable(tf.random_normal([env.hidden_size//2,1]))    
        self.Advantage = tf.matmul(self.streamA, self.AW)     
        self.Value = tf.matmul(self.streamV, self.VW)
        #Then combine them together to get our final Q-values.
        self.q_output = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
       
        self.predict = tf.argmax(self.q_output,1, name='predict') # vector of length batch size
        
        #Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.num_actions,dtype=tf.float32)
        
        # Importance sampling weights for PER, used in network update  xdg       
        self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)
        
        # select the Q values for the actions that would be selected         
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot), reduction_indices=1) # batch size x 1 vector
        
        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions      
        self.reg_vector = tf.maximum(tf.abs(self.Q)-env.REWARD_THRESHOLD,0)
        self.reg_term = tf.reduce_sum(self.reg_vector)
        self.abs_error = tf.abs(self.targetQ - self.Q)
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)
        
        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if env.per_flag:
            self.loss = tf.reduce_mean(self.per_error) + env.reg_lambda*self.reg_term
        else:
            self.loss = self.old_loss + env.reg_lambda*self.reg_term
            

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
        # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
            

def setHiddenLayer(state, hidden_size, phase, last_layer):
    if last_layer:
        fc = tf.contrib.layers.fully_connected(state, hidden_size, activation_fn=None)
    else:
        fc = tf.contrib.layers.fully_connected(state, hidden_size) 
    fc_bn = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=phase)
    fc_ac = tf.maximum(fc_bn, fc_bn*0.01)
    return fc, fc_bn, fc_ac    

def getValues(streamA, streamV, AW, VW):
    Advantage = tf.matmul(streamA,AW)
    Value = tf.matmul(streamV,VW)
    return Advantage, Value

def getErrors(tf, targetQ, imp_weights, q_output, actions_onehot, REWARD_THRESHOLD, reg_lambda):
    # select the Q values for the actions that would be selected         
    Q = tf.reduce_sum(tf.multiply(q_output, actions_onehot), reduction_indices=1) # batch size x 1 vector
        
    # regularisation penalises the network when it produces rewards that are above the
    # reward threshold, to ensure reasonable Q-value predictions  
    reg_vector = tf.maximum(tf.abs(Q)-REWARD_THRESHOLD,0)
    reg_term = tf.reduce_sum(reg_vector)
    abs_error = tf.abs(targetQ - Q)
    td_error = tf.square(targetQ - Q)
        
    # below is the loss when we are not using PER
    old_loss = tf.reduce_mean(td_error)
        
    # as in the paper, to get PER loss we weight the squared error by the importance weights
    per_error = tf.multiply(td_error, imp_weights)

    # total loss is a sum of PER loss and the regularisation term
    if env.per_flag:
        loss = tf.reduce_mean(per_error) + reg_lambda*reg_term
    else:
        loss = old_loss + reg_lambda*reg_term
    return imp_weights, Q, reg_vector, reg_term, abs_error, td_error, old_loss, per_error, loss 
 
            

def saveTestResult(env, df, ecr, save_dir, i):
    df.loc[:,'ECR']= np.nan
    df.loc[df.groupby(env.pid).head(1).index, 'ECR'] = ecr
    df, rdf = ld.rl_analysis_pdqn(env, df)
    df.to_csv(save_dir+"results/testdf_t"+str(i+1)+".csv", index=False) #h"+str(env.hidden_size)+"_
    return df, rdf


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
        self.keyword = args.k
        self.load_data = args.a
        self.character = args.c
        self.gamma = float(args.d)

        self.LEARNING_RATE = True # use it or not
        self.learnRate = float(args.l) # init_value (αk+1 = 0.98αk)
        self.learnRateFactor = 0.98
        self.learnRatePeriod = 5000
        self.belief = float(args.b)
        self.hidden_size = int(args.hu)
        self.numSteps = int(args.t)

        self.gpuID = str(args.g)
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
        
        self.DEBUG = False 
        self.targetTimeWindow = 0
        self.load_model = False #True
        self.save_results = True
        self.func_approx = 'LSTM' #'FC_S2' #'FC' 
        self.batch_size = 32
        self.period_save = 100000
        self.period_eval = 100000
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
        

import argparse
def load_policy(policy_dir, policyEpoch, polTDmode, feat):
    policyName = policy_dir+'models/regul'+str(policyEpoch)+'/'
    #print("policy:", policyName)
    # Load the policy network
    parser = argparse.ArgumentParser()
    polEnv = parsing(parser, polTDmode, feat)
    #polEnv = setGPU(tf, polEnv)
    #print("polEnv.stateFeat:", polEnv.stateFeat)
    mainQN, targetQN, saver, init = setNetwork(tf, polEnv)
    policy_sess = tf.Session(config=polEnv.config) 
    load_RLmodel(policy_sess, tf, policyName)
        
    return polEnv, policy_sess, mainQN, targetQN #, polTestdf

def parsing(parser, polTDmode, feat):   
    parser.add_argument("-a")# load_data
    parser.add_argument("-l")
    parser.add_argument("-t")    
    parser.add_argument("-g")   # GPU ID#
    parser.add_argument("-r")   # i: IR or DR
    parser.add_argument("-k")   # keyword for models & results
    parser.add_argument("-msl")  # max sequence length for LSTM
    parser.add_argument("-d")   # discount factor gamma

    parser.add_argument("-pb") # pred_val basis to distinguish pos from neg (0.5, 0.9, etc.)
    parser.add_argument("-c") # characteristics of model
    parser.add_argument("-b") # belief for dynamic TDQN
    parser.add_argument("-hu") # hidden_size

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
    polEnv.nextStateFeat = ['next_'+s for s in polEnv.stateFeat]
    polEnv.nextNumFeat = ['next_'+s for s in polEnv.numFeat]
    
    #print("nextNumFeat: ", env.nextNumFeat)   
    #print("nextstateFeat: ", env.nextStateFeat)
        
    return polEnv

def setNetwork(tf, env):
    tf.reset_default_graph()
    
    mainQN = RQnetwork(env, 'main')
    targetQN = RQnetwork(env, 'target')

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    return mainQN, targetQN, saver, init    

def load_RLmodel(sess, tf, save_dir):
    startTime = time.time()
    try: # load RL model
        restorer = tf.train.import_meta_graph(save_dir + 'ckpt.meta')
        restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
        print ("Model restoring time: {:.2f} sec".format((time.time()-startTime)))
    except IOError:
        print ("Error: No previous model found!") 


    
def getTargetQ(env, Q2, done_flags, actions_from_q1, rewards, tGammas):
    end_multiplier = 1 - done_flags # handles the case when a trajectory is finished
            
    # target Q value using Q values from target, and actions from main
    double_q_value = Q2[range(len(Q2)), actions_from_q1]

    # empirical hack to make the Q values never exceed the threshold - helps learning
    if env.Q_clipping:
        double_q_value[double_q_value > Q_THRESHOLD] = Q_THRESHOLD
        double_q_value[double_q_value < -Q_THRESHOLD] = -Q_THRESHOLD
    #print("double_q_value:", double_q_value)

    # definition of target Q
    if 'Expo' in env.character or 'Hyper' in env.character: # or 'TBD' in env.character:
        targetQ = rewards + (tGammas*double_q_value * end_multiplier)
    else:
        targetQ = rewards + (env.gamma*double_q_value * end_multiplier)    
    #print("targetQ:", targetQ)
    return targetQ




def process_train_batch(env, df, trainAll):
    
    if env.per_flag: # uses prioritised exp replay
        weights = df['prob']
    else:
        weights = None
        
    a = df.sample(n=env.batch_size, weights=weights) 
    
    idx = a.index.values.tolist()
        
    states, actions, rewards, next_states, next_actions, done_flags, tGammas = trainAll 
    states =  states[idx]
    actions = actions[idx] 
    next_actions = next_actions[idx]
    rewards = rewards[idx]
    next_states = next_states[idx]
    done_flags = done_flags[idx]
    if tGammas != []:
        tGammas = tGammas[idx]
    return states, actions, rewards, next_states, next_actions, done_flags, tGammas, a


def process_eval_batch(env, df, data, X, X_next):

    a = data.copy(deep=True)
    idx = a.index.values.tolist()
    actions = np.array(a.Action.tolist())
    next_actions = np.array(a.next_action.tolist()) 
    rewards = np.array(a[env.rewardFeat].tolist())
    done_flags = np.array(a.done.tolist())
    
    if 'lstm' in env.keyword: 
        states = np.array(ld.makeX_event_given_batch(df, a, env.stateFeat, env.pid, env.maxSeqLen))#X[idx]
        next_states = np.array(ld.makeX_event_given_batch(df, a, env.nextStateFeat, env.pid, env.maxSeqLen))#X_next[idx] 
        
    elif 'dqn' in env.keyword: # "DQN"
        states = X 
        next_states = X_next 
    
    if 'Expo' in env.character or 'Hyper' in env.character:
        tGammas = np.array(a.loc[:, env.discountFeat].tolist())
    else:
        tGammas = []
    return (states, actions, rewards, next_states, next_actions, done_flags, tGammas)   



def rl_learning(tf, env, df, val_df, save_dir, data):
    
    diffActionNum = 0
    diffActionRate = 0
    maxECR = 0
    bestECRepoch = 1
    bestShockRate = 1.0
    bestEpoch = 1
    notImproved = 0
    startTime = time.time()
    learnedFeats = ['target_action'] #'RecordID','target_q'
    save_path = save_dir+"ckpt"      #The path to save our model to.

    # The main training loop is here
    tf.reset_default_graph()
    
    mainQN = RQnetwork(env, 'main')
    targetQN = RQnetwork(env, 'target')
    

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    target_ops = ld.update_target_graph(trainables, env.tau)

    #with tf.Session(config=env.config) as sess:
    policySess = tf.Session(config=env.config) 
    
    df, env, log_df, maxECR, bestECRepoch, startIter = ld.initialize_model(env, policySess, save_dir, df, save_path, init) # load a model if it exists

    net_loss = 0.0

    trainAll, valAll = data 
    
    for i in range(startIter, env.numSteps):
        
        states, actions, rewards, next_states, next_actions, done_flags, tGammas, sampled_df = \
                            process_train_batch(env, df, trainAll)
        
        #print("state: {} next_state: {} ".format( np.shape(states), np.shape(next_states)))

        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = policySess.run(targetQN.q_output, 
                            feed_dict={targetQN.state:next_states,
                                       targetQN.phase:True, 
                                       mainQN.learning_rate:env.learnRate,
                                       targetQN.batch_size:env.batch_size})
        
        if 'sarsa' in env.character:
            targetQ = getTargetQ(env, Q2, done_flags, next_actions, rewards, tGammas) 
        else:
            # Run PDQN according to the prediction
            actions_from_q1 = policySess.run(mainQN.predict, feed_dict={mainQN.state:next_states, \
                     mainQN.phase:True, mainQN.learning_rate:env.learnRate, mainQN.batch_size:env.batch_size})
            targetQ = getTargetQ(env, Q2, done_flags, actions_from_q1, rewards, tGammas) 
            
        # Calculate the importance sampling weights for PER
        imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(df['imp_weight'])))
        imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
        imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001
        # print("imp_sampling_weights: {} - {}".format(np.shape(imp_sampling_weights), imp_sampling_weights))
        # Train with the batch

        _, loss, error = policySess.run([mainQN.update_model, 
                                         mainQN.loss, 
                                         mainQN.abs_error], \
                     feed_dict={mainQN.state: states, 
                                mainQN.targetQ: targetQ, 
                                mainQN.actions: actions, 
                                mainQN.phase: True, 
                                mainQN.imp_weights: imp_sampling_weights, 
                                mainQN.batch_size:env.batch_size,
                                mainQN.learning_rate:env.learnRate,})

        # ------------------------------------------------
        # Update target towards main network 
        if i % 1000 == 0:
            ld.update_target(target_ops, policySess)
        
        net_loss += sum(error)

        # Set the selection weight/prob to the abs prediction error 
        # and update the importance sampling weight
        new_weights = pow((error + env.per_epsilon), env.per_alpha)
        df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights
        df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0/len(df)) * (1.0/new_weights)), env.beta_start)

        #run an evaluation on the validation set
        if ((i+1) % env.period_eval==0) or i == 0: # evaluate the 1st iteration to check the initial condition 
            saver.save(policySess,save_path)
            av_loss = net_loss/(env.period_save * env.batch_size)          
            net_loss = 0.0
            #print ("Saving PER and importance weights")
            with open(save_dir + 'per_weights.p', 'wb') as f:
                pickle.dump(df['prob'], f)
            with open(save_dir + 'imp_weights.p', 'wb') as f:
                pickle.dump(df['imp_weight'], f)

            #1. Validation: ECR
            if env.DEBUG:
                print("DEBUG: Validation")
                print("policySess: {}, mainQN: {}, targetQN: {},\n val_df:{}, valAll: {}".format(policySess, mainQN, targetQN,
                                                                                  np.shape(val_df), np.shape(valAll)))    
            
                print("valAll len {}".format(np.shape(valAll[0])))
                
            val_df, ecr = ld.do_eval_pdqn_lstm(policySess, env, mainQN, targetQN, val_df, valAll)
                

            mean_abs_error = np.mean(val_df.error)
            mean_ecr = np.mean(ecr)
            avg_maxQ = val_df.groupby(env.pid).target_q.mean().mean() # mean maxQ by trajectory 
 
            if i+1 > env.period_eval and i > startIter+env.period_eval:
                curActions = val_df[['target_action']].copy(deep=True)
                diffActionNum = len(val_df[curActions['target_action'] != predActions['target_action']])
                diffActionRate = diffActionNum/len(curActions)

                if env.DEBUG:
                     print("diff: {}".format(val_df[curActions['target_action'] != 
                                                    predActions['target_action']].index))
            predActions = val_df[['target_action']].copy(deep=True)

            simPolicy = [] 
            
            if mean_ecr > maxECR and i > 1 :
                maxECR = mean_ecr
                bestECRepoch = i+1
                
            saveModel(save_dir, bestPath = save_dir+'models/pol_'+str(i+1)+'/')

            print("{}/{}/{}/h{}/g{}[{:.0f}] L:{:.2f}, Q:{:.2f}, E:{:.2f} (best: {:.2f} - {}),".
                  format(env.date, env.fold, env.character, env.hidden_size, env.gamma,(i+1),\
                                       av_loss, avg_maxQ, mean_ecr, maxECR, bestECRepoch),end=' ')
            print("act:{}({:.3f}) run time: {:.1f} m".format(diffActionNum, diffActionRate, 
                                                             (time.time()-startTime)/60))

            startTime = time.time()
            log_df.loc[len(log_df),:] = [i+1, av_loss, mean_abs_error, avg_maxQ, mean_ecr, env.learnRate,
                                         env.gamma, diffActionNum, diffActionRate]
            log_df.to_csv(save_dir+"results/log.csv", index=False)

        
    if startIter < env.numSteps:    
        saveModel(save_dir, bestPath = save_dir+'regul'+str(i+1)+'/')
        log_df.to_csv(save_dir+"results/log.csv", index=False)

    #Test & Analysis
    shockNum_all = []
    save_dir = save_dir+'eval/'+env.filename 
    print("Policy: {}".format(save_dir))
    
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    policySess.close 
    return df 

    

def setRanges(env, yhat):
    for i in range(len(env.labelFeat)):
        if (yhat[:, i] < env.labelFeat_min_std[i]) or (yhat[:, i] > env.labelFeat_max_std[i]):
            yhat[:, i] = env.labelFeat_min_std[i]
    return yhat

# Consider TD: Exclude the first TD, which tends to be long. (waiting time)
def applyTD(env, pdf, yhat, j):
    avgTD = ( pdf.loc[1:j,'TD'].median() + pdf.loc[1:j,'TD'].mean() )/ 2
    predicted = yhat[:, :len(env.numFeat)]
    current = np.array(pdf.loc[j, env.numFeat].values.tolist())
    yhat_final = current + (predicted-current)*pdf.loc[j, 'TD']/avgTD
    return yhat_final

def initSimulData(env, pdf):
    pdf.reset_index(inplace=True, drop=True)
    pdf.loc[:, env.actFeat] = pdf.loc[:, env.oaFeat].values.tolist() 
    
    tgTimeIdx = pdf[pdf.timeWindow==1].index.tolist()
    if len(tgTimeIdx) == 1:
        tgTimeIdx = tgTimeIdx*2

    firstMeasIdx = pdf[pd.notnull(pdf.SystolicBP_org)].head(env.minMeasNum).index
    if len(firstMeasIdx) == 0: # in case of minMeasNum == 0 (Use the original dataset)
        firstMeasIdx = [0]
        print("No measurement of systolicBP: vid = {}".format(e))     
        return pdf

    startIdx = np.max([firstMeasIdx[-1],tgTimeIdx[0]]) #gurantee the startIdx covers the minimum number of measurements
    startIdx = pdf[(pdf.index <= startIdx) & pd.notnull(pdf.SystolicBP_org)].tail(1).index[0] # guarntee the startIdx does not cheat the future value by backward filling.
    
    pdf.loc[tgTimeIdx[0], 'timeWindow'] = np.nan
    pdf.loc[startIdx, 'timeWindow'] = 1
    
    #if simulMode == 'policy':
    pdf.loc[:startIdx, 'target_action'] = pdf.loc[:startIdx, 'Action'] # set the original actions out of the window
    pdf.loc[startIdx:, env.actFeat] = 0 # rest the actions to 0 within the window
    pdf.loc[startIdx:, env.miFeat] = 0
    pdf.loc[startIdx:, 'VasoAdmin'] = np.nan
    pdf.loc[startIdx:, 'Anti_infective'] = np.nan
    return pdf, startIdx, tgTimeIdx


def carry_forward (data, hours, carry_list):
    for column in carry_list:
        #data[column+"_inf"] = np.nan
        data.loc[pd.notnull(data[column]), 'Ref_time'] = data.loc[:, 'MinutesFromArrival']
        data.loc[:,[column,'Ref_time']] = data.groupby(['VisitIdentifier'])[[column,'Ref_time']].ffill()
        data.loc[(data.MinutesFromArrival- data.Ref_time).round(6) > 60*hours, column] = np.nan 
        data.loc[:,'Ref_time'] = np.nan
    return data



def saveModel(save_dir, bestPath):
    if not os.path.exists(bestPath):
        os.makedirs(bestPath)
    
    shutil.copyfile(save_dir+'checkpoint', bestPath+'checkpoint')
    shutil.copyfile(save_dir+'ckpt.data-00000-of-00001', bestPath+'ckpt.data-00000-of-00001')
    shutil.copyfile(save_dir+'ckpt.index', bestPath+'ckpt.index')
    shutil.copyfile(save_dir+'ckpt.meta', bestPath+'ckpt.meta')
    shutil.copyfile(save_dir+'imp_weights.p', bestPath+'imp_weights.p')
    shutil.copyfile(save_dir+'per_weights.p', bestPath+'per_weights.p')
                    

def RLprocess(tf, env, df, val_df, hdf, data, saveFolder, model_dir): 
    if model_dir == '':
        model_dir = saveFolder+'/'+env.date+'/'+str(env.fold)+'/'+env.filename+'/'
        ld.createResultPaths(model_dir, env.date)
        
    hdf.to_csv(model_dir+'hyper_parameters.csv', index=False)
        
    if 'lstm' in env.keyword:   
        env.func_approx = 'LSTM'

    elif 'dqn' in env.keyword:
        env.func_approx = 'FC'

    print("Length: train({}), validation({})".format(len(df), len(val_df))) 
    _= rl_learning(tf, env, df, val_df, model_dir, data) 
        
    return model_dir
        

def getNextState(df, env, test_df, state):
    df[env.nextStateFeat] = df[env.stateFeat]
    df[env.nextStateFeat] = df.groupby(env.pid).shift(-1).fillna(0)[env.nextStateFeat]
    test_df[env.nextStateFeat] = test_df[env.stateFeat]
    test_df[env.nextStateFeat] = test_df.groupby(env.pid).shift(-1).fillna(0)[env.nextStateFeat]
    return df, test_df

