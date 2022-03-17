import numpy as np
from model_topology import model as m
input_ = np.random.rand(1000,10);
output_ = 0.2;
print(input_.shape)

model = m(input_.shape)
model.add(layer_type='fully_connected',neuron=5)
model.add(layer_type='sigmoid');
model.add(layer_type='fully_connected',neuron=5)
model.add(layer_type='sigmoid');
model.add(layer_type='fully_connected',neuron=5)
model.add(layer_type='sigmoid');
model.add(layer_type='fully_connected',neuron=5)
model.add(layer_type='sigmoid');
model.add(layer_type='fully_connected',neuron=3)
model.add(layer_type='sigmoid');
model.add(layer_type='fully_connected',neuron=1)
model.add(layer_type='sigmoid');

model.summary();

model.training(epoch=50,data_=input_,target=output_,loss='mse',lr='0.001')
output = model.predict(input_[0,:].reshape((1,10)))
print(output)

