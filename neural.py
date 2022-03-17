import numpy as np

class fully_connected:
	def __init__(self,previous_layer_size,neuron=1):
		[h,w]=previous_layer_size;
		self.neuron=neuron
		self.bias = np.ones((1,int(neuron)),dtype='uint8')
		self.weight=np.random.rand(int(w),int(neuron))
		print(self.weight.shape)
		self.delta=0
		self.bias = self.bias.astype('float32');
		self.output_size = self.bias.shape;
		self.output = 0;
		self.del_error_del_w = 0;

	def forward(self,previous_layer_activation):
		value = np.matmul(previous_layer_activation,self.weight) + self.bias
		self.output = value 
		return value

	def backward(self,sucessor,activation_grad):
		self.delta = np.matmul(sucessor.delta,sucessor.weight.transpose())*activation_grad
		#self.del_error_del_w =np.matmul(self.delta,self.output.transpose())

class sigmoid:
	def __init__(self,output_size):
		self.output_size= output_size
		self.output = 0;
		self.grad = 0;
	def forward(self,input_):
		self.grad = input_*(1-input_)
		self.output = 1/(1+np.exp(input_));
		return self.output;

class linear:
	def __init__(self,output_size):
		self.output_size= output_size
		self.output = 0;
		self.grad = 0;
	def forward(self,input_):
		self.grad = 1
		self.output = input_;
		return self.output;


class mean_sqaure_error:
	def __init__(self):
		self.grad=0;
	def mse(self,predict,target):
		self.grad = (predict-target)
		return np.power(predict-target,2.0)*0.5 		




