from stock_env_discrete import StockEnv
from stock_env_discrete import STOCK 
import random
import numpy as np

def run_stock():
    n_epoch = 100
    mean = 0
    for x in xrange(n_epoch):
        env = StockEnv(STOCK.Baidu)
        while True:
            action = random.randint(0, 5)
            observation_, reward, done = env.step(action)
            
            if done:
                if x == 0:
                    mean = env.asset - 10000
                else:
                    mean = 0.9 * mean + 0.1 * (env.asset - 10000)
                break
    print(mean)

    # end of game
    print('game over')


if __name__ == "__main__":
    run_stock()
