from stock_env_discrete import StockEnv
from stock_env_discrete import STOCK
from stock_env import STOCK as STOCKC
from hmm import StockHMM
from sklearn.externals import joblib
from RL_brain import DeepQNetwork
import numpy as np
import warnings
from matplotlib import pyplot as plt

# global variable
observation = None


def run_stock():
    warnings.filterwarnings("ignore")  # ignore warnings

    stockHMM = StockHMM(STOCKC.Baidu)
    # load model
    stockHMM.model = joblib.load("BaiDuHMM.pkl")

    mean = []
    for episode in range(200):
        env = StockEnv(STOCK.Baidu)
        env.set_count(999)
        step = 0

        # initialize observation
        global observation
        p_states = stockHMM.predict(1000)
        observation = [env.cash, env.n_stock, env.price_curr]
        observation = np.concatenate((observation, p_states))
        for x in range(1000):
            global observation

            # RL choose action based on observation
            action = RL.choose_action(np.array(observation, dtype=float))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            p_states = stockHMM.predict(x + 1001)
            observation_ = np.concatenate((observation_, p_states))

            RL.store_transition(np.array(observation, dtype=float), action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            step += 1

        # add to list
        mean.append(env.asset - 10000)

        if episode % 10 == 0:
            print('Episode %d' % episode)
        print(mean[-1])

    # calculate mean
    print(np.mean(mean), np.var(mean))
    plt.scatter(range(len(mean)), mean)
    plt.show()

    # end of game
    print('game over')


if __name__ == "__main__":
    RL = DeepQNetwork(11, 8,
                      learning_rate=0.005,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_stock()
