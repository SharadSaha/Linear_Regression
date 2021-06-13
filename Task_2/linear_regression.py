import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import dataset

class Linear_regression:

	def __init__(self,train_data,train_label):
		self.epochs=3000
		self.learning_rate=0.001
		self.train_label=train_label
		self.train_data=train_data
		self.W=np.zeros((3,1))
		# self.bias=np.ones((47,1))


	def plot(self):
		fig=plt.figure()
		ax=fig.add_subplot(111,projection='3d')
		# ax=fig.add_subplot(111,projection='3d')
		# ax.set_xlabel('Population of city (In 10,000s)')
		# ax.set_ylabel('Profit in $10,000s')
		ax.scatter(self.train_data[:,1],self.train_data[:,2],self.train_label,c='r',label="train_data")
		# ax.scatter(self.test_data,self.test_label,c='g',label="test_data")
		x1= np.linspace(-1,3,100)
		x2=np.linspace(-1,3,100)
		x1,x2= np.meshgrid(x1,x2)
		y = self.W[0][0]+self.W[1][0]*x1+self.W[2][0]*x2
		ax.plot_surface(x1,x2,y)
		ax.legend()
		plt.show()


	def normalize(self):
		mean1=np.mean(train_data[:,1])
		mean2=np.mean(train_data[:,2])
		mean3=np.mean(train_label[:,0])
		sd1=np.std(train_data[:,1])
		sd2=np.std(train_data[:,2])
		sd3=np.std(train_label[:,0])
		for i in range(47):
			self.train_data[i][1]=abs(self.train_data[i][1]-mean1)/sd1
			self.train_data[i][2]=abs(self.train_data[i][2]-mean2)/sd2
			self.train_label[i][0]=abs(self.train_label[i][0]-mean3)/sd3
		# print(self.train_data)

	def calculate(self,data):
		return np.dot(data,self.W)

	def loss(self):
		return (1/np.size(self.train_data))*np.sum(np.square(self.calculate(self.train_data)-self.train_label))

	def gradient(self):
		grad=np.dot(self.train_data.T,self.calculate(self.train_data)-self.train_label)
		return grad

	def train(self):
		for i in range(self.epochs):
			self.W-=self.gradient()*self.learning_rate
			print(self.loss())
			# self.bias+=self.learning_rate*(self.calculate(self.train_data)-self.train_label)

if __name__=="__main__":
	train_data,train_label=dataset.load_data('linear_regression')
	model=Linear_regression(train_data,train_label)
	model.normalize()
	model.train()
	model.plot()
	# print(model.train_data[:,1],model.train_data[:,2])
	# print(model.W)
