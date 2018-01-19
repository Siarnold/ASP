# HSRL: Hidden State-based Reinforcement Learning for Stocks

> As the big assignment for the course of Applied Stochastic Process in 2017 Fall, Tsinghua University   
> 19, Jan, 2018  
> Contact: siarnold@foxmail.com  
> Collaborator: Jiang LIU  

------

## Announcement

* All codes are released here under the GNU licenses as free documents. Anyone who promises the freedom of these documents is granted with the rights to read, use or modify these codes.

## Environment

* Python 3.5
* Modules: NumPy, hmmlearn, TensorFlow
* Run the program by ```python HSRL.py```

## Code Description

* ```hmm.py``` trains the StockHMM model with EM algorithm, and predicts the states with Viterbi algorithm
* ```hmm_trade.py``` uses the predicted states directly to empirically make transactions, as a baseline
* ```HSRL.py``` **our proposed model**, uses reinforcement learning to learn to maximize the rewards in a given stock market with a given amount of initial money, combining the hidden states predicted by StockHMM
* ```HSRL_new.py``` adds a hidden layer in the Q network, but not as good as the original HSRL
* ```Q_learning.py``` uses Q learning (a type of reinforcement learning) to make transactions, without the aid of StockHMM, as a baseline
* ```random_trade.py``` randomly makes stock transactions, as a baseline
* ```RL_brain.py``` defines the Q network (https://morvanzhou.github.io/)
* ```RL_brain_new.py``` adds a hidden layer for the original Q network 
* ```stock_env.py``` implements the Simple Stock Transaction Model (SSTM), -1 <= actions <= 1
* ```stock_env_discrete.py``` the discrete version of the SSTM, action = 0, 1, ..., 10
