# NAMAC-RL

The NAMAC (Nearly Autonomous Management and Control) system for advanced reactors aims to support diagnosis and prediction of reactors' states, autonomous decision making for control actions, and discrepancy checking between a target reactor and its digital twin.  
This code provides an reinforcement learning (RL) framework to train RL agents and evaluate induced RL policies to support a machine's autunomous decision making for control actions.

Abstract

In the NAMAC system, the reinforcement learning (RL) agent takes a role of inducing an optimal nuclear reactor control policy by interacting with the reactor environment. While deep reinforcement learning (DRL) offers many advantages, it rarely performs adequately when applied to real-world decision-making tasks, especially those involving irregular time series with sparse actions in a partially observable environment. Such data properties cause three main challenges: 1) temporal irregularity causes temporal errors and incorrect value estimation, 2) sparse actions make the agent difficult to grasp high-level states, and 3) partial observability leads to a learning bias of a single agent. To address these challenges, we propose a general Time-aware deep reinforcement learning framework that incorporates three methodologies: 1) Time-aware deep Q-Networks (TQN), which leverages time intervals to estimate states and expected return to handle temporal irregularity, and 2) Multi-Temporal Abstraction (MTA) mechanism, which abstracts temporal sequences in multi-temporal views to understand a high-level reactor state, and 3) Average-Q Networks (AQN), which averages multiple Q-values from a prediction model and concurrent policies to overcome a single agent’s learning bias in a partially observable environment. The proposed methods were validated against a standard deep Q-learning framework in a nuclear expert-designed accident case study. The results show that the proposed methods significantly outperform the standard deep Q-learning frameworks in the quality of nuclear reactor control policy. 



The code consists of two parts:

1. MTA-TQN (Multi-Temporal Abstraction with Time-aware deep Q-Networks)
 - To train a single agent policy, using DQN with two temopral functionalitis;
 (1) Time-aware deep Q-Networks (TQN) train RL policies with two types of time-awareness: time-aware state approximation with time interavls as state input (TState) and time-aware reward estimation with temporal discounting (TDiscount).  
 (2) Multi-Temporal Abstraction (MTA) trains RL policies using temporal abstraction mechanism which produces multiple views of temporal state representation for a single state.
 (3) MTA-TQN combines MTA with TQN.
 - From the experiment results, MTA-TQN and its variants (TQN, MTA-DQN, MTA-TQN) outperforms the baseline Deep Q-Networks (DQN)  in terms of reactor utility and stability of policy training. 

2. AQN (Average Q-Networks)
 - To average multiple agents policies to reduce bias of a single agent policy.
 - To be available three modes: 
 (1) AQN_P (predictive AQN) is to consider a current state and a future predictive state to decide a current action, 
 (2) AQN_C (concurrent AQN) is to consider multiple concurrent views of a current state, recognized from independentaly trained multi-agents policies.
 (3) AQN is to combine AQN_P and AQN_C. 
 - From the experiment results, AQN outperforms AQN_P, AQN_C, and other single agent policies in terms of reactor utility and stability of policy training. 


