import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import dataset


class Linear_regression:
	def __init__(self,train_data,train_label):
		self.epochs=2500
		self.learning_rate=0.01
		self.train_data=train_data
		self.train_label=train_label
		self.W=np.zeros((2,1))
		# self.bias=np.ones((97,1))
		


	def loss_func(self):
		loss=(1/(2*np.size(self.train_data)))*np.sum(np.square(self.calculate_result(self.train_data)-self.train_label))
		print("Loss:")
		return loss

	def gradient(self):
		grad=(1/97)*np.dot(self.train_data.T,self.calculate_result(self.train_data)-self.train_label)
		return grad


	def calculate_result(self,data):
		return np.dot(data,self.W)


	def train(self):
		# print("Initial loss:")
		# print(self.loss_func())
		for i in range(self.epochs):
			self.W-=self.learning_rate*self.gradient()
			# print(self.calculate_result(self.train_data))
			# self.bias+=self.learning_rate*np.sum(self.gradient())
			# print(self.bias)
			# print(self.loss_func())
		# print("The final weight is:")



	def plot(self):
		fig=plt.figure()
		# ax=fig.add_subplot(111,projection='3d')
		# ax.set_xlabel('Population of city (In 10,000s)')
		# ax.set_ylabel('Profit in $10,000s')
		plt.scatter(self.train_data[:,1],self.train_label,c='r',label="train_data")
		# ax.scatter(self.test_data,self.test_label,c='g',label="test_data")
		x = np.linspace(0,30,100)
		y = self.W[0][0]+self.W[1][0]*x
		plt.plot(x,y)
		plt.grid()
		plt.show()



if __name__=='__main__':
	train_data,train_label=dataset.load_data('task')
	model=Linear_regression(train_data,train_label)
	# print(model.train_data)
	# print(model.train_label)
	model.train()
	print(model.loss_func())
	print("The max profit is:")
	print(max(model.calculate_result(model.train_data)))
	model.plot()

	
	
