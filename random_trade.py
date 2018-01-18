from stock_env_discrete import StockEnv as EnvD
from stock_env import StockEnv as Env
from stock_env_discrete import STOCK as STOCKD
from stock_env import STOCK as STOCK
import random
import numpy as np


def run_stock_discrete():
    n_epoch = 100
    mean = []
    for x in range(n_epoch):
        env = EnvD(STOCKD.Baidu)
        env.set_count(999)  # from 1000 to 2000
        for i in range(1000):
            action = random.randint(0, 11)
            env.step(action)

        mean.append(env.asset - 10000)

        # while True:
        #     action = random.randint(0, 11)
        #     observation_, reward, done = env.step(action)
        #
        #     if done:
        #         mean.append(env.asset - 10000)
        #         break

    print(np.mean(mean), np.var(mean))
    # end of game
    print('game over')


def run_stock():
    n_epoch = 100
    mean = []
    for x in range(n_epoch):
        env = Env(STOCK.Baidu)
        env.set_count(999)  # from 1000 to 2000
        for i in range(1000):
            action = random.random() * 2 - 1  # [-1, 1]
            env.step(action)

        mean.append(env.asset - 10000)

        # while True:
        #     action = random.random() * 2 - 1  # [-1, 1]
        #     observation_, reward, done = env.step(action)
        #
        #     if done:
        #         mean.append(env.asset - 10000)
        #         break

    print(np.mean(mean), np.var(mean))
    # end of game
    print('game over')

if __name__ == "__main__":
    print('Continuous:')
    run_stock()
    print('Discrete:')
    run_stock_discrete()
