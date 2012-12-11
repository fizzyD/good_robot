# -*- coding: utf-8 -*-
""" 
The well behaved robot
======================================
From 
"""
import numpy as np
import neurolab as nl
#import numpy.random as rand

import pylab as pl
def plot_robot_judgement(nn_inputs, nn_outputs):
    # Plot results:
    pl.title('Training Data')
    childx = []
    childy = []
    adultx = []
    adulty = []
    vampirex = []
    vampirey = []
    werewolfx = []
    werewolfy = []
    for i in range(len(nn_outputs)):
        (tall, hairy) = nn_inputs[i]
        ( impale, scream, runaway, greet) = nn_outputs[i]
        if greet == 1 and scream == 1:
            childx.append(tall)
            childy.append(hairy)
        elif greet == 1 and scream ==0:
            adultx.append(tall)
            adulty.append(hairy)
        elif impale == 1:
            vampirex.append(tall)
            vampirey.append(hairy)
        elif runaway == 1:
            werewolfx.append(tall)
            werewolfy.append(hairy)
            pass
        print "[ impale, scream, runaway, greet]"
        print nn_outputs[i]
        pass
    pl.plot(childx,childy, 'bo')
    pl.hold(True)
    pl.plot(adultx,adulty, 'go')
    pl.plot(vampirex,vampirey, 'rd')
    pl.plot(werewolfx,werewolfy, 'yd')
    #pl.plot(adult, 'go')
    #pl.plot(vampire, 'rd')
    #pl.plot(werewolf, 'yd')
    pl.xlabel('Height')
    pl.ylabel('Hairiness')
    pl.legend(('child', 'adult','vampire','werewolf'))
    pl.show()
    pass

def main():
    net_savefile = 'robot_memory.txt'

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
    
    #Create net with 2 inputs and 4 neurons
    #net = nl.net.newc([[0.0, 1.0],[0.0, 1.0]], 4)
    #train with rule: Conscience Winner Take All algoritm (CWTA)
    #error = net.train(nn_inputs, epochs=200, show=20)

    plot_robot_judgement(nn_inputs, nn_outputs)

    size = len(heights)

    #create a network with 3 layers 2 inputs, 4 hidden, 4 outputs 
    #net = nl.net.newff([[min(heights), max(heights)],[min(hairiness),max(hairiness)]], [2, 4, 4, 4] )
    try:
        net = nl.load(net_savefile)
    except IOError:
        net = nl.net.newff([[min(heights), max(heights)],[min(hairiness),max(hairiness)]], [2, 8, 4] )
        pass
    
    # train the network
    if False:
        error = nl.train.train_rprop(net,np.array(nn_inputs), np.array(nn_outputs), epochs=10000, show=100, goal=0.2)
    #import pdb
    #pdb.set_trace()
        net.save(net_savefile)
        pass
    min_hairy = min(hairiness) 
    min_height = min(heights)
    tallrange = max(heights) - min_height -0.2
    hairyrange = max(hairiness) - min_hairy -0.2
    xpoints = 10
    ypoints = 10
    dx_height = float(tallrange)/xpoints
    dy_hairy = float(hairyrange)/ypoints
    dataset = []
    for ix in range(xpoints):
        for iy in range(ypoints):
            dataset.append([ix*dx_height+min_height,
                            iy*dy_hairy+min_hairy])
            pass
        pass
            
    out = net.sim(np.array(dataset))

    for i in range(len(out)):
        (a,b,c,d) = out[i]
        out[i] = [int(1) if a>0.5 else int(0),
                  int(1) if b>0.5 else int(0), 
                  int(1) if c>0.5 else int(0),
                  int(1) if d>0.5 else int(0)]
        pass

    plot_robot_judgement(dataset, out)
    while True:
        height = float(raw_input("Enter a height : "))
        hairy = float(raw_input("Enter hairiness : "))
        dataset = []
        dataset.append([height, hairy])
        input = np.array(dataset)
        print input
        out = net.sim(input)
        ( impale, scream, runaway, greet) = out[0]
        if greet > 0.5:
            print "Greetings..."
            pass
        if scream > 0.5:
            print "INTRUDER ALERT!"
            pass
        if impale > 0.5:
            print "STAKE TO THE HEART!"
            pass
        if runaway > 0.5:
            print "RUN AWAY! RUN AWAY!"
            pass
    #print out
    pass

if __name__ == "__main__":
    main()
    pass
