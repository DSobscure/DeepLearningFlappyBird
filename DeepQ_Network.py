#!/usr/bin/env python
from __future__ import print_function

import tensorflow as TensorFlow
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as Game
import random
import numpy as Numpy
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def CreateWeightVariable(shape):
    initial = TensorFlow.truncated_normal(shape, stddev = 0.01)
    return TensorFlow.Variable(initial)

def CreateBiasVariable(shape):
    initial = TensorFlow.constant(0.01, shape = shape)
    return TensorFlow.Variable(initial)

def Convolution2D(x, W, stride):
    return TensorFlow.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def MaxPool2x2(x):
    return TensorFlow.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def CreateNetwork():
    # network weights
    convolutionWeights1 = CreateWeightVariable([8, 8, 4, 32])
    convolutionBias1 = CreateBiasVariable([32])

    convolutionWeights2 = CreateWeightVariable([4, 4, 32, 64])
    convolutionBias2 = CreateBiasVariable([64])

    convolutionWeights3 = CreateWeightVariable([3, 3, 64, 64])
    convolutionBias3 = CreateBiasVariable([64])

    fullyConnectedWeights1 = CreateWeightVariable([1600, 512])
    fullyConnectedBias1 = CreateBiasVariable([512])

    fullyConnectedWeights2 = CreateWeightVariable([512, ACTIONS])
    fullyConnectedBias2 = CreateBiasVariable([ACTIONS])

    # input layer
    inputStates = TensorFlow.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    convolutionHiddenLayer1 = TensorFlow.nn.relu(Convolution2D(inputStates, convolutionWeights1, 4) + convolutionBias1)
    hiddenLayer1Pool = MaxPool2x2(convolutionHiddenLayer1)

    convolutionHiddenLayer2 = TensorFlow.nn.relu(Convolution2D(hiddenLayer1Pool, convolutionWeights2, 2) + convolutionBias2)
    #h_pool2 = max_pool_2x2(h_conv2)

    convolutionHiddenLayer3 = TensorFlow.nn.relu(Convolution2D(convolutionHiddenLayer2, convolutionWeights3, 1) + convolutionBias3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    reshapedConvolutionHiddenLayer3 = TensorFlow.reshape(convolutionHiddenLayer3, [-1, 1600])

    fullyConnectedHiddenLayer1 = TensorFlow.nn.relu(TensorFlow.matmul(reshapedConvolutionHiddenLayer3, fullyConnectedWeights1) + fullyConnectedBias1)

    # readout layer
    outputs = TensorFlow.matmul(fullyConnectedHiddenLayer1, fullyConnectedWeights2) + fullyConnectedBias2

    return inputStates, outputs, fullyConnectedHiddenLayer1

def TrainNetwork(inputStates, outputs, network, interactiveSession):
    # define the cost function
    a = TensorFlow.placeholder("float", [None, ACTIONS])
    y = TensorFlow.placeholder("float", [None])
    valueOutputs = TensorFlow.reduce_sum(TensorFlow.multiply(outputs, a), reduction_indices=1)
    cost = TensorFlow.reduce_mean(TensorFlow.square(y - valueOutputs))
    train_step = TensorFlow.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    gameState = Game.GameState()

    # store the previous observations in replay memory
    gameRecordDqueue = deque();
    gameRecord = deque();
    scores = deque();
    currentScore = 0.;
    
    globalMax = 0;
    gameCount = 1;

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = Numpy.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = gameState.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    currentStates = Numpy.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    #saver = TensorFlow.train.Saver()
    interactiveSession.run(TensorFlow.global_variables_initializer())
    saver = TensorFlow.train.Saver()
    saver.restore(interactiveSession, "SavedNetwork/InitialModel.ckpt")
    print("Model restored.")

    # start training
    t = 0
    
    while True:
        # choose an action epsilon greedily
        outputValues = outputs.eval(feed_dict={inputStates : [currentStates]})[0]
        actions = Numpy.zeros([ACTIONS])
        actionIndex = 0
        if t % FRAME_PER_ACTION == 0:
            actionIndex = Numpy.argmax(outputValues)
            actions[actionIndex] = 1
        else:
            actions[0] = 1 # do nothing

        # run the selected action and observe next state and reward
        coloredAfterState, reward, terminal = gameState.frame_step(actions)
        afterState = cv2.cvtColor(cv2.resize(coloredAfterState, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, afterState = cv2.threshold(afterState, 1, 255, cv2.THRESH_BINARY)
        afterState = Numpy.reshape(afterState, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        afterStates = Numpy.append(afterState, currentStates[:, :, :3], axis=2)
        reward = int(reward)
        currentScore += reward

        # store the transition in Dequeue
        gameRecord.append((currentStates, actions, reward, afterStates, terminal))

        if terminal:
            averageScore = currentScore / len(gameRecord)
            print("Game", gameCount, " score: ", currentScore, "average score: ", averageScore, " t: ", t);
            with open('result.txt', 'a') as file:
                file.writelines(str(currentScore) + "\n")
            for i in range(0, len(gameRecord)):
                x = gameRecord.popleft()
                sameRecord = (x[0], x[1], averageScore, x[3], x[4])
                gameRecordDqueue.append(sameRecord);
                if len(gameRecordDqueue) > REPLAY_MEMORY:
                    gameRecordDqueue.popleft()
            scores.append(currentScore);
            if len(scores) > 100:
                scores.popleft();
            if currentScore > globalMax:
                globalMax = currentScore
            currentScore = 0;
            if gameCount % 1000 == 0:
                savePath = saver.save(interactiveSession, "SavedNetwork/model.ckpt")
                print("Model saved in file: %s" % savePath)
            if gameCount % 100 == 0:
                print("=========================");
                print("AverageScore: ", Numpy.mean(numpyScores), ", Deviation: ", Numpy.std(numpyScores));
                print("MaxScore: ", Numpy.max(scores));
                print("GlobalMaxScore: ", globalMax);
                print("=========================");
            gameCount = gameCount + 1;

        if t > OBSERVE:
            # sample a minibatch to train on
            batch = random.sample(gameRecordDqueue, BATCH)

            # get the batch variables
            currentStatesBatch = [d[0] for d in batch]
            actionsBatch = [d[1] for d in batch]
            rewardBatch = [d[2] for d in batch]
            afterStateBatch = [d[3] for d in batch]

            valueBatch = []
            afterStateBatchOutput = outputs.eval(feed_dict = {inputStates : afterStateBatch})
            for i in range(0, len(batch)):
                terminal = batch[i][4]
                # if terminal, only equals reward
                if terminal:
                    valueBatch.append(-1)
                else:
                    valueBatch.append(rewardBatch[i] + GAMMA * Numpy.max(afterStateBatchOutput[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : valueBatch,
                a : actionsBatch,
                inputStates : currentStatesBatch}
            )

        # update the old values
        currentStates = afterStates
        t += 1

def PlayGame():
    interactiveSession = TensorFlow.InteractiveSession()
    inputStates, outputs, network = CreateNetwork()
    TrainNetwork(inputStates, outputs, network, interactiveSession)

def main():
    PlayGame()

if __name__ == "__main__":
    main()
