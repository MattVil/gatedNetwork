import os
import cv2
import numpy as np
import sys
import random
import time
import math
import pickle
import matplotlib
import datetime
import matplotlib.pyplot as plt

import MLP2 as MLP2
import mlp as MLP
import Dataset
from utils import record_video, plot_error_evolution, convertGray
from utils import get_center_frame, matMultiplication

plot = True
saveNetwork = True

session = "sess8"
if not os.path.exists("output/{}".format(session)):
    os.makedirs("output/{}".format(session))

lrate = 0.01
momentum = 0.0

sizeX = 40
sizeY = 40
hidden = 500
bottleneck = 150

def erreurQuadratique(reel, attendu):
    return ((reel-attendu)**2).sum()

def training(network, x1, x2):
    x1 = cv2.resize(x1, (sizeX, sizeY))
    x2 = cv2.resize(x2, (sizeX, sizeY))

    x = matMultiplication(x1, x2)

    x_flat = np.reshape(x, (sizeX*sizeY))

    output = network.propagate_forward(x_flat)
    mse = erreurQuadratique(output, x_flat)
    network.propagate_backward(x_flat, lrate, momentum)

    return mse, x, output

def test(network, x1, x2):
    x1 = cv2.resize(x1, (sizeX, sizeY))
    x2 = cv2.resize(x2, (sizeX, sizeY))

    x = matMultiplication(x1, x2)

    x_flat = np.reshape(x, (sizeX*sizeY))

    output = network.propagate_forward(x_flat)
    mse = erreurQuadratique(output, x_flat)

    return mse, x, output

def plot_weights(network, position, save_path):

    weights = network.getWeights()
    nb_filter = len(weights[position][0])
    input_size = len(weights[position])-1 # subtract the bias

    filters = []
    for i in range(nb_filter):
        side_size = (int(round(math.sqrt(input_size-1))))
        filter = np.zeros(side_size**2, np.float32)
        for j in range(input_size):
            # print("{}/{}".format(i, j))
            filter[j] = weights[position][j][i]
        filters.append(np.reshape(filter, (side_size, side_size)))

    fig=plt.figure(figsize=(8, 8))
    columns = math.sqrt(nb_filter)+1
    rows = math.sqrt(nb_filter)
    for i in range(1, nb_filter+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(filters[i-1])
    plt.show()
    plt.savefig("output/{}/weights_{}.png".format(session, datetime.datetime.now()))

def plot_in_out(in_img, out_img):
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(in_img, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(out_img, cmap='gray')

    now = datetime.datetime.now()
    fig.savefig('output/{}/{}.png'.format(session, now))
    plt.close(fig)

def learnDataset(dataset, mlp, repeat=1):
    """ For our experiment, param correspond to the number of hidden neurone """
    network = mlp

    errs = []
    idx = 0

    print("\n################# TRAINING #################\n")
    for x_train, y_train in zip(dataset.x_train, dataset.y_train):
        time1 = time.time()
        err, mult, output = training(network, x_train, y_train)
        errs.append(err)
        time2 = time.time()
        duration = (time2-time1)*1000.0
        if idx%100 == 0:
            print("n°:{0}\t\tErr: {1:.3f}\t\tTime: {2:.3f}ms".format(idx,
                                                                  err,
                                                                  duration))
            plot_in_out(mult, np.reshape(output, (sizeX, sizeY)))

        idx += 1

    if plot:
        # plot_weights(network, 1, "r")
        plot_error_evolution(errs, "output/{}/evol.png".format(session))
    if saveNetwork:
        print("Saving ... ", end="")
        network.save("networks/network{}.pckl".format(datetime.datetime.now()))
        print("Done.")
    return errs

def testDataset(dataset, network):
    errs = []
    idx = 0
    layer_activity = []

    print("\n################### TEST ###################\n")
    for x_test, y_test in zip(dataset.x_test, dataset.y_test):
        time1 = time.time()
        err, mult, output = test(network, x_test, y_test)
        errs.append(err)
        time2 = time.time()
        duration = (time2-time1)*1000.0
        layer_activity.append(network.getLayer()[1])
        if idx%10 == 0:
            print("n°:{0}\t\tErr: {1:.3f}\t\tTime: {2:.3f}ms".format(idx,
                                                                  err,
                                                                  duration))
            plot_in_out(mult, np.reshape(output, (sizeX, sizeY)))

        idx += 1
    layer_activity = np.asarray(layer_activity)

    print(layer_activity.shape)

    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 1, 1)
    plt.imshow(layer_activity.transpose())
    plt.colorbar()
    plt.grid()
    fig.savefig('activity.png')   # save the figure to file
    plt.close(fig)

def main():

    mlp = MLP.MLP(sizeX*sizeY, bottleneck, sizeX*sizeY)
    # mlp.load("networks/network2018-11-08 17:07:20.342072.pckl")
    dataset = Dataset.Dataset((sizeX, sizeY))
    #dataset.build("../dataset/lrud_001.txt", from_vid="../dataset/full-07_11_2019-001.avi")
    dataset.load_dataset("../dataset/lrud_001.txt")
    # dataset.plot_dataset(537, 538)
    err = learnDataset(dataset, mlp)

    dataset2 = Dataset.Dataset((sizeX, sizeY))
    dataset2.load_dataset("../dataset/test.txt")
    testDataset(dataset2, mlp)

if __name__ == '__main__':
    main()
