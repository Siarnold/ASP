from enum import Enum
from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import datetime
import os

STOCK = Enum('Stock', ('Google', 'Tencent', 'Baidu'))


	
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
		data,self.dates= self.get_data(path=path)
		self.open = data[:,0]
		self.high = data[:,1]
		self.low = data[:,2]
		self.close = data[:,3] 
		self.adj_close = data[:,4]
		self.volume = data[:,5]
		#number of hidden state
		self.n = 5
	def get_data(self,path):
		f = open(path)
		lines = f.readlines()
		f.close()

		# the first line is the header
		lines = lines[1:]

		x = []
		dates = []
		for line in lines:
			data = np.double(line.split(',')[1:7]);
			dates.append(line.split(',')[0]);
			x.append(data)  # [1] is the opening price

		return np.array(x),np.array(dates)
	def train(self):
		ld_hl = np.log(self.high) - np.log(self.close) # log difference of high and low 
		ld_c5 = np.log(self.close[5:]) - np.log(self.close[:-5]) # log difference of close (every 5 days)
		ld_c1 = np.log(self.close) # log of close
		ld_v5 = np.log(self.volume[5:]) - np.log(self.volume[:-5])
		ld_hl = ld_hl[5:]
		ld_c1 = ld_c1[4:]
		close = self.close[5:]
		dates = self.dates[5:]
		features = np.column_stack([ld_hl,ld_c5,ld_v5]) # stack features
		model = GaussianHMM(n_components= self.n, covariance_type="full", n_iter=2000).fit(features) # predict HMM models
		self.hidden_states = model.predict(features)  # predict hidden states (from data #5)
		plt.figure(figsize=(25, 18)) 
		axes = np.arange(0,close.shape[0])
		for i in range(model.n_components):
			pos = (hidden_states==i)
			plt.plot(axes[pos],close[pos],'o',label='hidden state %d'%i,lw=2)
			plt.legend(loc="left")
		plt.show()
		os.system("pause")
if __name__ == '__main__':
	stockhmm = StockHMM(STOCK.Baidu)
	stockhmm.train()
	
	