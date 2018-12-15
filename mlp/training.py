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

# MAIN HYPERPARAMETRES
plot = True
saveNetwork = True
session = "sess45"
if not os.path.exists("output/{}".format(session)):
    os.makedirs("output/{}".format(session))

# PATHS
path_to_dataset = "../dataset/matthieu2.txt"
# path_to_dataset = "../dataset/test.txt"
path_to_video = ""
path_weights = ""
# path_weights = "./networks/networksess24.pckl"
# path_weights = "./networks/networksess32.pckl"

# NETWORK HYPERPARAMETRES
lrate = 0.1
lr_min = 0.01
momentum = 0.00
sizeX = 80
sizeY = 80
hidden = 0
bottleneck = 25
nb_epochs = 5

# DATASET HYPERPARAMETRES
delta_time = 5
test_ratio = 0.3
valid_ratio = 0.1

def erreurQuadratique(reel, attendu):
    return ((reel-attendu)**2).sum()

def step_decay(iter, drop_rate=1000):
    initial_lrate = lrate
    drop = 0.5
    epochs_drop = drop_rate
    lr = initial_lrate * math.pow(drop, math.floor((1+iter)/epochs_drop))
    if(lr < lr_min):
        lr = lr_min
    return lr

def training(network, x1, x2, lr):
    x1 = cv2.resize(x1, (sizeX, sizeY))
    x2 = cv2.resize(x2, (sizeX, sizeY))

    x = matMultiplication(x1, x2)

    x_flat = np.reshape(x, (sizeX*sizeY))

    output = network.propagate_forward(x_flat)
    mse = erreurQuadratique(output, x_flat)
    network.propagate_backward(x_flat, lr, momentum)

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

def plot_in_out(in_img, out_img, folder):
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(in_img, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(out_img, cmap='gray')

    now = datetime.datetime.now()
    fig.savefig('{}/{}.png'.format(folder, now))
    plt.close(fig)

def plot_bar_activity(activity):
    fig, ax = plt.subplots()
    index = np.arange(bottleneck)
    bar_width = 0.35
    opacity = 0.4
    rects1 = ax.bar(index, activity, bar_width,
                    alpha=opacity, color='b',
                    label='Activity')
    fig.tight_layout()
    now = datetime.datetime.now()
    fig.savefig('output/{}/test/{}.png'.format(session, now))
    plt.close(fig)

def learnDataset(network, X, Y, X_valid, Y_valid, epochs=1):

    errs, valid = [], []
    idx = 0
    nb_iter = epochs*X.shape[0]

    if not os.path.exists("output/{}/training".format(session)):
        os.makedirs("output/{}/training".format(session))

    # Compute learning rate decay
    diff = lrate - lr_min
    lr_delta = diff/nb_iter
    lr = lrate
    iter_max = epochs*X.shape[0]

    print("\n################# TRAINING #################\n")
    for i in range(epochs):
        for x_train, y_train in zip(X, Y):
            time1 = time.time()
            # lr = step_decay(idx, 3000)
            err, mult, output = training(network, x_train, y_train, lr)
            errs.append(err)
            time2 = time.time()
            duration = (time2-time1)*1000.0
            if idx%100 == 0:
                err_valid = validDataset(network, X_valid, Y_valid)
                valid.append(err_valid)
                print("{0}\t\tErr: {1:.3f}\tValid_err: {2:.3f}\tTime: {3:.3f}ms\tLr: {4:.4f}".format(idx, err, err_valid, duration, lr))
                plot_in_out(mult,
                            np.reshape(output, (sizeX, sizeY)),
                            "output/{}/training".format(session))

            idx += 1
            lr -= lr_delta

    if plot:
        plot_weights(network, 0, "r")
        plot_error_evolution(errs, "output/{}/evol.png".format(session))
        plot_error_evolution(valid, "output/{}/evolValid.png".format(session))

    if saveNetwork:
        print("Saving ... ", end="")
        path_weights = "networks/network{}.pckl".format(session)
        network.save(path_weights)
        print("Done.")

    return errs

def validDataset(network, X, Y):
    errs = []
    idx = 0

    for x_valid, y_valid in zip(X, Y):
        time1 = time.time()
        err, mult, output = test(network, x_valid, y_valid)
        errs.append(err)
        time2 = time.time()
        duration = (time2-time1)*1000.0
        idx += 1
    return (sum(errs)/len(errs))

def testDataset(network, X, Y):
    errs = []
    idx = 0
    layer_activity = []

    if not os.path.exists("output/{}/test".format(session)):
        os.makedirs("output/{}/test".format(session))

    print("\n################### TEST ###################\n")
    for x_test, y_test in zip(X, Y):
        time1 = time.time()
        err, mult, output = test(network, x_test, y_test)
        errs.append(err)
        time2 = time.time()
        duration = (time2-time1)*1000.0
        midValue = network.getLayer()[1]
        # print(midValue)
        layer_activity.append(midValue)
        if idx%50 == 0:
            print("n°:{0}\t\tErr: {1:.3f}\t\tTime: {2:.3f}ms".format(idx,
                                                                  err,
                                                                  duration))
            plot_in_out(mult,
                        np.reshape(output, (sizeX, sizeY)),
                        "output/{}/test".format(session))

        idx += 1
    layer_activity = np.asarray(layer_activity)

    plot_error_evolution(errs, "output/{}/evolTest.png".format(session))

    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 1, 1)
    plt.imshow(layer_activity.transpose())
    plt.colorbar()
    plt.grid()
    fig.savefig('output/{}/activity.png'.format(session))   # save the figure to file
    plt.close(fig)

def test_activity(network, X, Y):
    errs = []
    idx = 0
    layer_activity = []

    if not os.path.exists("output/{}/test".format(session)):
        os.makedirs("output/{}/test".format(session))

    err, mult, output = test(network, X[0], Y[0])
    initial_activity = network.getLayer()[1].copy()

    print("\n############### TEST ACTIVITY ###############\n")
    for x_test, y_test in zip(X, Y):
        time1 = time.time()

        # Pass the test Image
        err, mult, output = test(network, x_test, y_test)
        errs.append(err)

        time2 = time.time()
        duration = (time2-time1)*1000.0

        # Monitor Layer activity
        current_activity = network.getLayer()[1]
        diff_activity = (current_activity - initial_activity)
        layer_activity.append(diff_activity)

        if idx%30 == 0:
            print("n°:{0}\t\tErr: {1:.3f}\t\tTime: {2:.3f}ms".format(idx,
                                                                     err,
                                                                     duration))

            # Plot the Input/Output Image
            plot_in_out(mult,
                        np.reshape(output, (sizeX, sizeY)),
                        "output/{}/test".format(session))

            # Plot the layer activity
            plot_bar_activity(diff_activity)

        idx += 1

    layer_activity = np.asarray(layer_activity)

    # Plot the neurons activity during time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(bottleneck):
        ax.plot(layer_activity[:,i], label=i)
    fig.savefig('output/{}/neuron_activity.png'.format(session))
    plt.show()

    # Plot the neurons similarity
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    neurones = []
    for i in range(bottleneck):
        neuron = layer_activity[:, i]
        less_activ = np.argsort(neuron)
        neurones.append(less_activ)
        ax2.plot(less_activ, label=i)
    fig2.savefig('output/{}/neuron_similarity.png'.format(session))
    plt.show()


    neurones = np.asarray(neurones)
    plt.imshow(np.transpose(layer_activity))
    plt.colorbar()
    plt.show()


    plot_error_evolution(errs, "output/{}/evolTestActivity.png".format(session))

def save_session_parameters():
    with open("output/{}/parameters.txt".format(session),"w+") as f:
        f.write("SESSION : {}\n".format(session))
        f.write("------------------------------------------------\n")
        f.write("Network :\n")
        f.write("\tINPUT : {}x{}\n".format(sizeX, sizeY))
        f.write("\tHIDDEN : {}\n".format(bottleneck))
        f.write("\tLEARNING RATE : {} to {}\n".format(lrate, lr_min))
        f.write("\tEPOCHS : {}\n".format(nb_epochs))
        f.write("\tMOMENTUM : {}\n".format(momentum))
        f.write("\tWEIGHTS INIT FROM : {}\n".format(path_weights))
        f.write("\tPATH TO WEIGHTS : {}\n".format("networks/network{}.pckl".format(session)))
        f.write("------------------------------------------------\n")
        f.write("Dataset : \n")
        f.write("\tDELTA TIME : {}\n".format(delta_time))
        f.write("\tORIGINAL VIDEO : {}\n".format(path_to_video))
        f.write("\tTEST RATIO : {}\n".format(test_ratio))
        f.write("\tVALID RATIO : {}\n".format(valid_ratio))
        f.write("\tPATH TO DATASET : {}\n".format(path_to_dataset))


def main():

    mlp = MLP.MLP(sizeX*sizeY, bottleneck, sizeX*sizeY)
    if(path_weights != ""):
        mlp.load(path_weights)
        print("Weights init from {}".format(path_weights))

    dataset = Dataset.Dataset((sizeX, sizeY))
    # dataset.build("../dataset/matthieu2.txt", from_vid="../dataset/full-07_11_2019-001.avi")
    # dataset.build("../dataset/mb.txt", from_vid="../dataset/bout.avi")
    # dataset.build("../dataset/mbl.txt", from_vid="../dataset/louis.avi")
    # dataset.build("../dataset/mblr.txt", from_vid="../dataset/remi.avi")
    dataset.load(path_to_dataset)
    x_train, y_train, x_test, y_test, x_valid, y_valid = dataset.split_and_process(delta_time,
                                                                                   test_ratio,
                                                                                   valid_ratio)

    err = learnDataset(mlp, x_train, y_train, x_valid, y_valid, nb_epochs)

    datasetTest = Dataset.Dataset((sizeX, sizeY))
    datasetTest.load("../dataset/test.txt")
    x, y = datasetTest.process(5)
    test_activity(mlp, x, y)

    save_session_parameters()

if __name__ == '__main__':
    main()
