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

more options:  -b 0.1 -tf 200 -d 0.97 -hu 128 -t 200000 -msl 5 -g 3 -cvf 0 -g 0
  - b: belief for the temporal discount function
  - tf: task time window for the temporal discount function
  - d : constant discount
  - hu: number of hidden units for deep function approximation
  - t: max training update
  - msl: max sequence length for LSTM/dense
  - g: GPU ID
  - cvf: fold id for a different random seed
========================================================================================
