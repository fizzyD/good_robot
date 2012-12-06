# -*- coding: utf-8 -*-
""" 
Example of use competitive layer(newc)
======================================

"""

import numpy as np
import neurolab as nl
#import numpy.random as rand

file = open('werewolf_data1.txt','r')

nn_inputs = []
heights = []
hairiness = []
nn_inputs = []
nn_outputs = []
for line in file.readlines():
    (i1,i2,o1,o2,o3,o4) = line.split()
    heights.append(float(i1))
    hairiness.append(float(i2))
    nn_inputs.append([float(i1),float(i2)])
    nn_outputs.append([int(o1), int(o2), int(o3), int(o4)])
    pass

# Create net with 2 inputs and 4 neurons
#net = nl.net.newc([[0.0, 1.0],[0.0, 1.0]], 4)
# train with rule: Conscience Winner Take All algoritm (CWTA)
#error = net.train(nn_inputs, epochs=200, show=20)

# Plot results:
import pylab as pl
pl.title('Training Data')
pl.plot(heights, hairiness,'x')
pl.show()

size = len(heights)

# create a network with 3 layers 2 inputs, 4 hidden, 4 outputs 
#net = nl.net.newff([[min(heights), max(heights)],[min(hairiness),max(hairiness)]], [2, 4, 4, 4] )
net = nl.net.newff([[min(heights), max(heights)],[min(hairiness),max(hairiness)]], [2, 8, 4] )

# train the network
error = nl.train.train_rprop(net,np.array(nn_inputs), np.array(nn_outputs), epochs=50000, show=100, goal=0.02)
#import pdb
#pdb.set_trace()
out = net.sim(np.array(nn_inputs))
