from hmmlearn.hmm import GaussianHMM
import numpy as np
from stock_env import plot_hidden_states, STOCK
from sklearn.externals import joblib


# HMM model used for stock price prediction
class StockHMM:
    def __init__(self, stock=STOCK.Google):
        if stock == STOCK.Google:
            path = './data/GOOG.csv'
        elif stock == STOCK.Baidu:
            path = './data/BIDU.csv'
        elif stock == STOCK.Tencent:
            path = './data/TCEHY.csv'
        else:
            print('Invalid argument!')
            raise SystemError()

        # initialize data
        data, self.dates = self.get_data(path=path)
        self.open = data[:, 0]
        self.high = data[:, 1]
        self.low = data[:, 2]
        self.close = data[:, 3]
        self.adj_close = data[:, 4]
        self.volume = data[:, 5]  # the number of stocks in stock transactions per day
        self.model = None
        
    # read data
    def get_data(self, path):
        f = open(path)
        lines = f.readlines()
        f.close()
        # the first line is the header
        lines = lines[1:]
        x = []
        dates = []
        for line in lines:
            data = np.double(line.split(',')[1:7])
            dates.append(line.split(',')[0])
            x.append(data)  # [1] is the opening price
        return np.array(x), np.array(dates)

    # train model with nc hidden states from the first n (including) data
    def train(self, nc, n):
        features = self.features_extraction(n)
        self.model = GaussianHMM(n_components=nc, covariance_type="full", n_iter=2000).fit(features) # predict HMM models
    
    # extract features from first n (not including) data
    def features_extraction(self, n):
        assert 5 < n < self.high.shape[0]

        ld_hl = np.log(self.high) - np.log(self.low)  # log difference of high and low
        ld_c5 = np.log(self.close[5:n]) - np.log(self.close[:n-5])  # log difference of close (every 5 days)
        ld_v5 = np.log(self.volume[5:n]) - np.log(self.volume[:n-5])
        ld_hl = ld_hl[5:n]
        # concatenate to form features
        features = np.column_stack([ld_hl, ld_c5, ld_v5])  # dim: (n-5) * 3
        return features

    # predict the states of the nth period
    def predict(self, n):
        features = self.features_extraction(n - 1)
        hidden_states_proba = self.model.predict_proba(features)
        states = hidden_states_proba[-1, :]

        return states.dot(self.model.transmat_)


if __name__ == '__main__':
    nc = 5  # number of hidden states
    stock_hmm = StockHMM(STOCK.Google)

    # train with first 1000 data
    # stock_hmm.train(nc, 1000)
    stock_hmm.model = joblib.load("baiduHMM.pkl")

    x = []
    # test, predict states from 1000 to 2000
    for i in range(1000, 2000):
        x.append(stock_hmm.predict(i))
    x = np.array(x)

    states = x.argmax(axis=1)

    # plot
    plot_hidden_states(states, stock_hmm.close[1000:2000], nc)

    # save the model
    # joblib.dump(stock_hmm.model, "HMM.pkl")
