import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#df=pd.DataFrame(x_train)
# file=df.to_csv('file.csv')
# display(df)


class Linear_regression:
  def __init__(self,x_train,y_train,x_test,y_test):
    self.epochs=32000
    self.train_data=tf.cast(x_train,dtype=tf.float32)
    self.train_label=tf.cast(y_train,dtype=tf.float32)
    self.test_data=tf.cast(x_test,dtype=tf.float32)
    self.test_label=tf.cast(y_test,dtype=tf.float32)
    self.weights=tf.zeros(shape=[13,1],dtype=tf.float32)
    self.bias=1
    self.W=tf.Variable(self.weights,dtype=tf.float32)
    self.B=tf.Variable(self.bias,dtype=tf.float32)
    self.optimiser=tf.keras.optimizers.Adam(learning_rate=0.1)
    # print(self.train_data.shape,self.train_label.shape)


  def calc(self,inp_data):
    return tf.matmul(inp_data,self.W)+self.B


  @tf.function
  def update(self,inp_data,label):
    loss_func=lambda:tf.reduce_mean((self.calc(inp_data)-label)**2)
    self.optimiser.minimize(loss_func,[self.W,self.B])

  def train(self):
    for i in range(self.epochs):
      self.update(self.train_data,self.train_label)
      if i%100==0:
        print("\rloss {}".format(tf.sqrt(tf.reduce_mean((self.calc(self.test_data)-self.test_label)**2))),end='')
        sys.stdout.flush()
    
  def frame(self,inp_data,label):
    df=pd.DataFrame(inp_data)
    display(df)

  # def normalize(self,train_data):
  #   mean=[]
  #   std=[]
  #   for i in range(13):
  #     mean.append(np.mean(train_data[:,i]))
  #     std.append(np.std(train_data[:,i]))
  #   for i in range(404):
  #     for j in range(13):
  #       # print(i,j)
  #       train_data[i][j]=abs(train_data[i][j]-mean[j])/std[j]
  #   return train_data
      
# def reduce(self,inp_data):
  # m=tf.

#def init_params(weights,bias):

def init_model_params():
  pass
  
if __name__=="__main__":
  (x_train,y_train),(x_test,y_test)=tf.keras.datasets.boston_housing.load_data(path='boston_housing.npz', test_split=0.2, seed=113)
  x_train=tf.reshape(x_train,shape=[404,13])
  y_train=tf.reshape(y_train,shape=[-1,1])

  # x_test=tf.reshape(x_test,shape=[404,13])
  y_test=tf.reshape(y_test,shape=[-1,1])
  model=Linear_regression(x_train,y_train,x_test,y_test)

  # x_train=tf.keras.utils.normalize(x_train,axis=-1,order=2)
  # y_train=tf.keras.utils.normalize(y_train,axis=-1,order=2)

  # x_test=tf.keras.utils.normalize(x_test,axis=-1,order=2)
  # y_test=tf.keras.utils.normalize(y_test,axis=-1,order=2)
  # x_train=np.array(x_train)
  # y_train=np.array(y_train)
  # y_train=[(y_train[i]-np.mean(y_train))/np.std(y_train) for i in range(404)]
  # print(x_train.shape)
  # x_train=model.normalize(x_train)
  # print(np.stack(x_train[:,:9],x_train[:,10:]),axis=0)
  # x_train=pd.DataFrame(x_train)
  # model.frame(x_train,y_train)
  # print(np.stack(x_train.iloc[:,:model.frame(x_train,y_train)9],x_train.iloc[:,10:]),axis=0)

  # x_train=tf.convert_to_tensor(x_train)
  # y_train=tf.convert_to_tensor(y_train)

  # x_test=tf.convert_to_tensor(x_test)
  # y_test=tf.convert_to_tensor(y_test)


  covar=tf.linalg.matmul(tf.transpose(x_train),x_train)
  e,v=tf.linalg.eigh(covar)
  plt.plot(e)
  plt.show()
  #print(y_train)

  model.train()
