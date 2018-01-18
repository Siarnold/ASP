from hmm import StockHMM
from stock_env import StockEnv
from stock_env import STOCK 
from stock_env_discrete import StockEnv as StockEnvD
from stock_env_discrete import STOCK as STOCKD
from sklearn.externals import joblib
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # ignore warnings

    stockHMM = StockHMM(STOCK.Baidu)
    # load model
    stockHMM.model = joblib.load("BaiDuHMM.pkl")

    print('Continuous: ')
    env = StockEnv(STOCK.Baidu)
    # [1000, 2000)
    env.set_count(999)
    for x in range(1000):
        p_states = stockHMM.predict(x + 1000)
        # order: 2 3 4 1 0
        my_action = p_states[2] + p_states[3] * 0.5 - p_states[1] * 0.5 - p_states[0]
        env.step(my_action)

    print(env.asset - 10000)

    print('Discrete: ')
    env = StockEnvD(STOCKD.Baidu)
    # [1000, 2000)
    env.set_count(999)
    for x in range(1000):
        p_states = stockHMM.predict(x + 1000)
        # order: 2 3 4 1 0
        my_action = round(5 * (p_states[2] + p_states[3] * 0.5 - p_states[1] * 0.5 - p_states[0]) + 5)
        env.step(my_action)

    print(env.asset - 10000)

