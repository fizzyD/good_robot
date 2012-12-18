# -*- coding: utf-8 -*-
""" 
The well behaved robot
======================================
Scott Dyer
Neural Net Project for Aritificial Intelligence

This code was tested with python 2.6 and requires the neurolab
package as well as numpy.

This projects was based on the course from http://en.wikiversity.org/wiki/Learning_and_neural_networks
The goal of this project was to use the same data set but implement it in python using a neural net
package and use the numpy plotting to enable visualization of the network classification. neurolab 
was selected as it had a concise set of examples that also showed training error as a function of 
training iterations. The examples were simple, easy to understand and demonstrated a number of neural
network configurations and training techniques.

From the wikiversity page:
Your great uncle Otto recently passed away leaving you his mansion in Transylvania. 
When you go to move in, the locals warn you about the Werewolves and Vampires that lurk in the area. 
They also mention that both vampires and werewolves like to play pool, which is alarming to you since 
your new mansion has a billiard room. Being a savvy computer scientist you come up with a creative solution. 
You will buy a robot from Acme Robotics (of Walla Walla Washington). You’re going to use the robot to guard 
your billiard room and make sure nothing super natural finds its way there.


"""
import numpy as np
import neurolab as nl
#import numpy.random as rand

import pylab as pl

def plot_robot_judgement(nn_inputs, nn_outputs):
"""
Plots the robot judgements as a function of the inputs
"""
    # Plot results:
    pl.title('Training Data')
    # the arrays (x,y) for each class of being
    childx = []
    childy = []
    adultx = []
    adulty = []
    vampirex = []
    vampirey = []
    werewolfx = []
    werewolfy = []
    # loop over the neuralnet outputs
    # get the associate inputs
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
        #show the action table
        print nn_outputs[i]
        pass
    # create a plot of what was identified as a child
    pl.plot(childx,childy, 'bo')
    # keep the above data visible
    pl.hold(True)
    # add a plot of what was identified as an adult
    pl.plot(adultx,adulty, 'go')
    # add a plot of what was identified as a vampire
    pl.plot(vampirex,vampirey, 'rd')
    # add a plot of what was identified as a werewolf
    pl.plot(werewolfx,werewolfy, 'yd')
    #pl.plot(adult, 'go')
    #pl.plot(vampire, 'rd')
    #pl.plot(werewolf, 'yd')
    pl.xlabel('Height')
    pl.ylabel('Hairiness')
    pl.legend(('child', 'adult','vampire','werewolf'))
    pl.show() # probably want to see it...
    pass

def main():
"""
Main function, reads in training data, train neural net, displays results,
shows coverage of the network and finishes the user input for a demo.
"""
    # the file name of the trained neuralnet
    net_savefile = 'robot_memory.txt'
    # file name of the training data for the neural net
    file = open('werewolf_data1.txt','r')

    nn_inputs = []
    heights = []
    hairiness = []
    nn_inputs = []
    nn_outputs = []
    # go through each training line in the net to create the training data
    for line in file.readlines():
        #Data format
        #[height,hairiness,impale, scream, runaway, greet]
        (i1,i2,o1,o2,o3,o4) = line.split()
        heights.append(float(i1))
        hairiness.append(float(i2))
        nn_inputs.append([float(i1),float(i2)])
        # action data is in the following sequence [ impale, scream, runaway, greet]
        nn_outputs.append([int(o1), int(o2), int(o3), int(o4)])
        pass
    
    #Create net with 2 inputs and 4 neurons,
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
    if False: # this is turned off since the net is trained...
        # would be better to add a command line argues whether to turn on training  --train 
        error = nl.train.train_rprop(net,np.array(nn_inputs), np.array(nn_outputs), epochs=10000, show=100, goal=0.2)
    #import pdb
    #pdb.set_trace()
        net.save(net_savefile)
        pass

    #This is debug code to look at the net to see how it classifies, basically map the net.
    min_hairy = min(hairiness) 
    min_height = min(heights)
    tallrange = max(heights) - min_height -0.2
    hairyrange = max(hairiness) - min_hairy -0.2
    xpoints = 10 # for finer resoluton increase these values
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
    # the net needs has to be quantized for display purposes
    # but needs to be differentiable for training.
    for i in range(len(out)):
        (a,b,c,d) = out[i]
        out[i] = [int(1) if a>0.5 else int(0),
                  int(1) if b>0.5 else int(0), 
                  int(1) if c>0.5 else int(0),
                  int(1) if d>0.5 else int(0)]
        pass
    
    plot_robot_judgement(dataset, out)
    # Finally is the demo mode, we have made the plots,
    # let the user test different values.
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
    """
    Check to see if we are the main program,
    If so do our thing.
    """
    main()
    pass
