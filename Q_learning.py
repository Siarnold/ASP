from stock_env_discrete import StockEnv
from stock_env_discrete import STOCK 
from RL_brain import DeepQNetwork
import numpy as np

def run_stock():
    env = StockEnv(STOCK.Baidu)
    mean = 0
    for episode in range(50):
    	step = 0
        while True:
            observation = [env.cash, env.n_stock, env.price_curr]

            # RL choose action based on observation
            action = RL.choose_action(np.array(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(np.array(observation), action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                if episode == 0:
                    mean = env.asset - 10000
                else:
                    mean = 0.9 * mean + 0.1 * (env.asset - 10000)
                print(mean)
                env = StockEnv(STOCK.Baidu)
                break
            step += 1

    print(mean)
    # end of game
    print('game over')


if __name__ == "__main__":
    RL = DeepQNetwork(5, 3,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_stock()
