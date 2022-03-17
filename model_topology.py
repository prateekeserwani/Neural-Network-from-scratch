import numpy as np
from neural import fully_connected as fc
from neural import sigmoid as sg
from neural import linear as li
from neural import mean_sqaure_error as mm


class model:
	def __init__(self,size):
		self.input_size = size; 	
		self.list=list()
		self.layers_name = list();
		self.layers_name.append('INPUT DATA');
		self.layers_type = list()

	def add(self,layer_type,neuron=1):
		if layer_type=='fully_connected':
			if len(self.list)==0:
				layer = fc(self.input_size,neuron)		
				self.layers_name.append('FC');
			else:
				len_ = len(self.list)
				layer = fc(self.list[len_-1].output_size,neuron)		
				self.layers_name.append('FC');
			self.layers_type.append(1)

		if layer_type=='sigmoid' or layer_type=='linear':
			len_ = len(self.list)

			if layer_type=='sigmoid':		
				layer = sg(self.list[len_-1].output_size)		
				self.layers_name.append('SIGMOID');
			if layer_type=='linear':
				layer = li(self.list[len_-1].output_size)		
				self.layers_name.append('LINEAR');
			self.layers_type.append(0)
		self.list.append(layer);

	def summary(self):
		print("summary of model")
		for i in range(len(self.layers_name)):
			if i==0:
				print(self.layers_name[i], self.input_size)
			else:
				if self.layers_type[i-1] == 1:
					print(self.layers_name[i], self.list[i-1].weight.shape)
				else:
					print(self.layers_name[i])
		

	def predict(self,input_):
		for i in range(len(self.list)):
			if i==0:
				output = self.list[i].forward(input_);
			else:
				output = self.list[i].forward(output)
		return output


	
	def weight_update(self,data,lr):	
		feature = data;
		for i in range(len(self.list)):
			if self.layers_type[i]==1:
				self.list[i].del_error_del_w =np.matmul(feature.transpose(),self.list[i].delta)
				self.list[i].weight = self.list[i].weight - float(lr)* self.list[i].del_error_del_w		
			feature = self.list[i].output

				

	def training(self,epoch,data_,target,loss,lr):
		iteration=0;
		for ee in range(epoch):
			for sample in range(1000):
				data = data_[sample,:];
				data = data.reshape(1,10);
				if loss == 'mse':
					matric = mm();
				pred = self.predict(data)
				loss = matric.mse(pred,target)
				flag=True
				last_weight_layer=0;
		
				for i in reversed(range(len(self.list))):
		
					if self.layers_type[i] == 0:
						grad_activation = self.list[i].grad
		
					if self.layers_type[i]==1:
						if flag==True:
							self.list[i].delta = matric.grad*grad_activation
							last_weight_layer=i;
							flag=False;
						else:
							self.list[i].backward(self.list[last_weight_layer],grad_activation);
							last_weight_layer=i;
				self.weight_update(data,lr)
				iteration = iteration + 1;
				print("iteration = " + str(iteration) +"loss = " + str(loss) + "predicted " + str(pred))
		
		
