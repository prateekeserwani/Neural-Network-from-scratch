import numpy as np

class activation_function:
	def __init__(self,output_size):
		self.output_size= output_size
		self.grad = 0;
	def sigmoid(self,input_):
		self.grad = input_*(1-input)
		return 1/(1+np.exp(input_));


