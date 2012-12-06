import numpy as np
import neurolab as nl

input = np.random.uniform(0, 0.1, (1000, 225))
output = input[:,:10] + input[:,10:20]
# 2 layers with 225 inputs 50 neurons in hidden\input layer and 10 in output
# for 3 layers use some thet: nl.net.newff([[0, .1]]*225, [50, 40, 10])
net = nl.net.newff([[0, .1]]*225, [50, 10])
net.trainf = nl.train.train_bfgs

e = net.train(input, output, show=1, epochs=100, goal=0.0001)

