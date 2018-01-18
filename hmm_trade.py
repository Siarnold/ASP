from hmm import StockHMM
from stock_env import StockEnv
from stock_env import STOCK 
from sklearn.externals import joblib

if __name__ == '__main__':
    env = StockEnv(STOCK.Baidu)
    env.count = 1500;
    nc = 5; #number of hidden states
    Baidu = StockHMM("Baidu")
    # load model
    Baidu.model = joblib.load("BaiDuHMM.pkl")
    # Trade 300 periods
    for x in range(env.count,env.count+400):
        states = Baidu.predict(x)
        # 0 2 4 for up ; 1 3 down
        my_action = -0.1*(states[0] + states[2] + states[4]) + 0.05
        print('New info: ((cash, n_stock, current price), reward, done)')
        print(env.step(my_action))
    print('You have earned $%f just now.' % (env.asset - 10000))
