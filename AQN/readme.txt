====================================================================================
Instructions for Average Q-Network (AQN)
* AQN can be implemented in the test/evaluation time, if pre-trained policies are ready.

* Execution file (Evaluation): eval_AQN.py
   
Average Q-Network (AQN), which combines 
 1) Predictive AQN (AQNp)
 2) Concurrent AQN (AQNc) 

* Prerequisites --------------------------------------------------------------------------
For predictive AQN (AQNp), 
1. Load an LSTM-based simulator for forcasting future observations (to estimate predicitve 
    Q-values in AQNp)
2. Set the mode AQNp:  
     for future Q-only AQN, the predicted time list (predN) has only one future time 
        predN = [future time1] (e.g. predN = [120] : 120-sec future Q estimation).
     for current and future Q-average, the predicted time list (predN) includes 
        predN = [0, future time1, future time2, ...] (e.g. predN = [0, 60])

For concurrent AQN (AQNc),
1. Load Multiple pre-trained policies (either using 'RQN" or TQN') should be placed in 
   the designated folders (we used 10 pre-trained policies for each model)
2. Without temporal average (AQNp), the predicted time list (predN) has only one 
    current time, predN = [0]. / With AQNp, use its predN. 
3. For evaluation, N-action timing, 'tgTimeList', should be randomly selected and fixed for 
   a fair comparison. (N depends on a task.)


* Test 
 $ python eval_AQN.py -method={RQN, TQN} -i=0 -f=0

 -method: RQN for AQN, TQN for TAQN
 -f: first_policyID for AQNc 
   - e.g. When the number of policies is 3,
      - if first_policyID is set to 0, the averaged policies will be [pol0, pol1, pol2]
      - if first_policyID is set to 9, the averaged policies will be [pol9, pol0, pol1]
 -i : start iteration for evaluation 
    default: set to 0 
    if the evaluation result file exists, it can be continued with the next iteration 

Simplified Algorithm ---------------------------------------------------------------   
1. Select multiple pre-trained policies for AQNc 
2. Given states of the test set, calculate and average multiple Q-values from 
   the selected policies
3. Calculate the average utility of test set with the recommended actions
4. loop 2-3 for each combination of multiple policies.
