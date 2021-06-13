import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd

class Dataset:

	def __init__(self):
		self.df=pd.read_csv("ex1data2.txt")
		self.val1=np.resize(self.df['x1'].to_numpy(),(47,1))
		self.val2=np.resize(self.df['x2'].to_numpy(),(47,1))
		self.train_label=np.resize(self.df['y'].to_numpy(),(47,1))
		self.val=np.ones((47,1))
		self.train_data=np.column_stack((self.val1,self.val2))
		self.train_data=np.column_stack((self.val,self.train_data))


	def plot(self):
		print(x.train_data)
		fig=plt.figure()
		ax=fig.add_subplot(111,projection='3d')
		ax.set_xlabel('Population of city (In 10,000s)')
		ax.set_ylabel('Profit in $10,000s')
		ax.scatter(self.val1,self.val2,self.train_label,c='r',label="train_data")
		# ax.scatter(self.test_data,self.test_label,c='g',label="test_data")
		ax.legend()
		plt.show()


def load_data(data):
	if data=="linear_regression":
		x=Dataset()
		return x.train_data,x.train_label

if __name__=="__main__":
	x=Dataset()
	x.plot()
