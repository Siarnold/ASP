from enum import Enum
from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import datetime
	
class StockHMM():
	def __init__(self, stock = "Google"):
		if stock == "Google":
			path = './data/GOOG.csv'
		elif stock == "Baidu":
			path = './data/BIDU.csv'
		elif stock == "Tencent":
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
		
	# read data
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
		
	# plot hidden states with close price
	def plot(hidden_states,close,nc):
		plt.figure(figsize=(25, 18)) 
		axes = np.arange(0,close.shape[0])
		for i in range(nc):
			pos = (hidden_states==i)
			plt.plot(axes[pos],close[pos],'o',label='hidden state %d'%i,lw=2)
			plt.legend(loc="best")
		plt.show()
		
	# train model with nc hidden states from first m data 
	def train(self,nc,n):
		features = self.features_extraction(n);
		self.model = GaussianHMM(n_components = nc, covariance_type="full", n_iter=2000).fit(features) # predict HMM models
		self.transm = self.model.transmat_
	
	# extract features from first n data
	def features_extraction(self,n):
		assert 4<n<self.high.shape[0]
		ld_hl = np.log(self.high) - np.log(self.close) # log difference of high and low 
		ld_c5 = np.log(self.close[5:n]) - np.log(self.close[:n-5]) # log difference of close (every 5 days)
		ld_v5 = np.log(self.volume[5:n]) - np.log(self.volume[:n-5])
		ld_hl = ld_hl[5:n]
		features = np.column_stack([ld_hl,ld_c5,ld_v5]) # stack features
		return features
	# predict the states of #n period
	def predict(self,n):
		features = self.features_extraction(n-1)
		hidden_states_proba = self.model.predict_proba(features)
		states = hidden_states_proba[n-7,:]
		return states.dot(self.transm)
		
	

if __name__ == '__main__':
	nc = 5; #number of hidden states
	Baidu = StockHMM("Baidu")
	# train with first 1000 data
	Baidu.train(nc,1000)
	x = []
	# test, predic states from 1000 to 2000
	for i in range(1000,2000):
		x.append(Baidu.predict(i))
	x = np.array(x)
	states = x.argmax(axis = 1)
	# plot
	StockHMM.plot(states,Baidu.close[1000:2000],nc)