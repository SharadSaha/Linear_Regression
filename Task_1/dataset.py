import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

class Dataset:
	def __init__(self):
		self.df1=pd.read_csv("ex1data1.txt")
		self.train_data=np.resize(self.df1['x'].to_numpy(),(97,1))
		self.train_label=np.resize(self.df1['y'].to_numpy(),(97,1))
		self.val=np.ones((97,1))
		self.train_data=np.column_stack((self.val,self.train_data))
		# Y represents profit and X represents population (both in 10000s)
		# self.df2=pd.read_csv("ex1data2.txt")
		# self.test_data=np.resize(self.df2['x'].to_numpy(),(97,1))
		# self.test_label=np.resize(self.df2['y'].to_numpy(),(97,1))


	def plot(self):
		fig=plt.figure()
		ax=fig.add_subplot(111,projection='3d')
		ax.set_xlabel('Population of city (In 10,000s)')
		ax.set_ylabel('Profit in $10,000s')
		ax.scatter(self.train_data[:,1],self.train_label,c='r',label="train_data")
		# ax.scatter(self.test_data,self.test_label,c='g',label="test_data")
		ax.legend()
		plt.show()


def load_data(info):
	if info=='task':
		data=Dataset()
		return data.train_data,data.train_label



if __name__=="__main__":
	data=Dataset()
	print(data.train_data)
	# print(data.test_data,data.test_label)
	data.plot()
