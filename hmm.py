from enum import Enum
from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime


STOCK = Enum('Stock', ('Google', 'Tencent', 'Baidu'))

def get_data(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    # the first line is the header
    lines = lines[1:]

    x = []
    for line in lines:
        data = np.double(line.split(',')[1:7]);
        x.append(data)  # [1] is the opening price

    return np.array(x)
	
class StockHMM():
	def __init__(self, stock = STOCK.Google):
		if stock == STOCK.Google:
			path = './data/GOOG.csv'
		elif stock == STOCK.Baidu:
			path = './data/BIDU.csv'
		elif stock == STOCK.Tencent:
			path = './data/TCEHY.csv'
		else:
			print('Invalid argument!')
			raise SystemError()
		#set up data
		data = get_data(path=path)
		self.open = data[:,0]
		self.high = data[:,1]
		self.low = data[:,2]
		self.close = data[:,3] 
		self.adj_close = data[:,4]
		self.volume = data[:,5]
		#number of hidden state
		self.n = 6
	def train(self):
		ld_hl = np.log(self.high) - np.log(self.close) # log difference of high and low 
		ld_c5 = np.log(self.close[5:]) - np.log(self.close[:-5]) # log difference of close (every 5 days)
		ld_c1 = np.log(self.close) # log of close
		ld_v5 = np.log(self.volume[5:]) - np.log(self.volume[:-5])
		ld_hl = ld_hl[5:]
		ld_c1 = ld_c1[4:]
		features = np.column_stack([ld_hl,ld_c5,ld_v5])
		model = GaussianHMM(n_components= self.n, covariance_type="full", n_iter=2000).fit([features])
		hidden_states = model.predict(features)
		plt.figure(figsize=(25, 18)) 
		for i in range(model.n_components):
			pos = (hidden_states==i)
			plt.plot_date(Date[pos],close[pos],'o',label='hidden state %d'%i,lw=2)
			plt.legend(loc="left")
if __name__ == '__main__':
	stockhmm = StockHMM(STOCK.Google)
	stockhmm.train()

	