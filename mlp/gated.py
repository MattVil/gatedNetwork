import cv2
import numpy as np
import sys
import random
import time
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt

import MLP2 as MLP
import Dataset
from utils import record_video, plot_error_evolution, convertGray
from utils import get_center_frame, matMultiplication


sizeX = 40
sizeY = 40

debutX = 160
debutY = 240

pas = 15000

lrate = 0.01
momentum = 0.000
cumulateur = 0

coucheCache = 50

cluster = 50

nbIteration = 10
sizeMiniLot = 25
repeat = 1

plot = False
saveNetwork = True

frame_delay = 5 #on prend une frame sur frame_delay

vid_out = "../dataset/test-08_11_2019-001.avi"
vid_in = "long_1-5.avi"

path_save_dataset = "dataset/medium.txt"
path_to_dataset = "dataset/medium.txt"

#Definive the state of the programe

activateChargementNetwork = False
activateSparseCoding = False

def erreurQuadratique(reel, attendu):
    return ((reel-attendu)**2).sum()

def training_3imgs(network, x1, x2, x3):

    x = matMultiplication(x1, x2)

    x_flat = np.reshape(x, (sizeX*sizeY))
    x3_flat = np.reshape(x3, (sizeX*sizeY))

    output = network.propagate_forward(x_flat)
    mse = erreurQuadratique(output, x3_flat)
    network.propagate_backward(x3_flat, lrate, momentum)

    return x, mse

def training(network, x1, x2):

    x = matMultiplication(x1, x2)

    x_flat = np.reshape(x, (sizeX*sizeY))

    output = network.propagate_forward(x_flat)
    mse = erreurQuadratique(output, x_flat)
    network.propagate_backward(x_flat, lrate, momentum)

    return mse, x

def training(network, x):
    x_flat = np.reshape(x, (sizeX*sizeY))

    output = network.propagate_forward(x_flat)
    # output += 1
    # output /= 2
    mse = erreurQuadratique(output, x_flat)
    network.propagate_backward(x_flat, lrate, momentum)

    return mse, output

def live_learning(from_vid=False, plot=False, frame_delay=1):
    network = MLP.MLP(sizeX*sizeY, coucheCache, sizeX*sizeY)

    if(from_vid):
        cap = cv2.VideoCapture(vid_in)
    else:
        cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Unable to connect to camera.")
        return 1

    x1 = np.zeros((480,640,1), np.uint8)
    x2 = np.zeros((480,640,1), np.uint8)

    ret, frame = cap.read()
    frame_gray = convertGray(frame)
    prev_gray = frame_gray

    errs = []
    points = ((0, 0), (0, 0))
    firstTime = True
    n_img, n = 0, 0
    while cap.isOpened():


        ret, frame = cap.read()
        frame_gray = convertGray(frame)

        if ret and n%frame_delay==0:
            previous = prev_gray
            prev_gray = frame_gray

            prev_face, points = get_center_frame(previous, (sizeX, sizeY))
            frame_face, points = get_center_frame(frame_gray, (sizeX, sizeY))

            mult_face = matMultiplication(prev_face, frame_face)

            cv2.imshow("Multiplication", cv2.resize(mult_face, (300, 300)))

            cv2.rectangle(frame, points[0], points[1], (0,255,0), 3)
            time1 = time.time()
            err, output = training(network, mult_face)
            print(output)
            cv2.imshow("Output", cv2.resize(output, (300, 300)))
            errs.append(err)
            n_img += 1
            time2 = time.time()
            duration = (time2-time1)*1000.0
            show_layer_activation(network, 1)
            if n_img%100 == 0:
                print("n°: {0}\t\tErr: {1:.3f}\t\tTime: {2:.3f}ms".format(n_img,
                                                                      err,
                                                                      duration))

        cv2.rectangle(frame, points[0], points[1], (0,255,0), 3)
        cv2.imshow("Live", frame)

        n += 1
        if(n_img >= 995 and from_vid):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # plot the error curve
    if plot:
        plot_error_evolution(errs, "new")

    return errs

def plot_weights(network, save_path):

    weights = network.getWeights()
    nb_filter = len(weights[0][0])
    input_size = len(weights[0])-1 # subtract the bias

    filters = []
    for i in range(nb_filter):
        side_size = (int(round(math.sqrt(input_size-1))))
        filter = np.zeros(side_size**2, np.float32)
        for j in range(input_size):
            # print("{}/{}".format(i, j))
            filter[j] = weights[0][j][i]
        filters.append(np.reshape(filter, (side_size, side_size)))

    fig=plt.figure(figsize=(8, 8))
    columns = math.sqrt(nb_filter)+1
    rows = math.sqrt(nb_filter)
    for i in range(1, nb_filter+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(filters[i-1])
    plt.show()

def show_layer_activation(network, layer):
    height = 500
    width = 500
    confidence = 0.8
    activity = network.getLayer()[layer]
    delta = (0.2 * height) // len(activity)
    bar_size = (0.8 * height) // len(activity)

    image = np.zeros((height+10,width+120,3), np.uint8)
    image[:,:,:] = (255, 255, 255)

    for i in range(1, len(activity)+1):
        if(activity[i-1] > 0):
            lenght = int(width * activity[i-1]) - 50
        else:
            lenght = int(width * -activity[i-1]) - 50
        bottom_left = int((i*delta + (i-1)*bar_size))
        top_right = int((i*delta + i*bar_size))
        if(activity[i-1] > 0):
            cv2.rectangle(image, (0, bottom_left), (lenght, top_right), (141, 214, 88), -1)
        else:
            cv2.rectangle(image, (0, bottom_left), (lenght, top_right), (60, 73, 231), -1)

        cv2.putText(image,"Neuron {}".format(i-1),(height+150, bottom_left+15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        value = activity[i-1]*100
        cv2.putText(image,"{:.2f}".format(value),(lenght-60, bottom_left+15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imshow("Activity", image)


def learnDataset(dataset, param=None, repeat=1, plot=False):
    """ For our experiment, param correspond to the number of hidden neurone """
    network = MLP.MLP(sizeX*sizeY, param, sizeX*sizeY)

    errs = []
    for i in range(repeat):
        for idx, img in enumerate(dataset.data):
            time1 = time.time()
            err, output = training(network, img)
            errs.append(err)
            time2 = time.time()
            duration = (time2-time1)*1000.0
            if idx%100 == 0:

                # img2 = (img+1)/2
                # output2 = np.reshape((output+1)/2, (sizeX,sizeY))
                # print("{}\tMin : {}\tMax : {}".format(img2.shape, np.amin(output2), np.amax(output2)))
                # cv2.imshow("Exemple Input", cv2.resize(img2, (250,250)))
                # cv2.imshow("Exemple Output", cv2.resize(output2, (250,250)))
                print("n°:{0}\t\tErr: {1:.3f}\t\tTime: {2:.3f}ms".format(idx,
                                                                      err,
                                                                      duration))
    if plot:
        plot_weights(network, "r")
        plot_error_evolution(errs, "learnDataset")
    if saveNetwork:
        print("Saving ... ", end="")
        network.save("networks/network{}.pckl".format(param))
        print("Done.")
    return errs

def benchmarkHiddenSize(queue, figure_name):

    dataset = Dataset.Dataset((sizeX, sizeY))
    dataset.load_dataset(path_to_dataset)
    legend = []
    for i in queue:
        print("#"*80)
        print("\t\t\tHidden size : {}".format(i))
        print("#"*80)
        err = learnDataset(dataset, i)
        print("\nLast err = {}".format(err[-1]))
        plt.plot(err)
        legend.append('N_h = {}'.format(i))

    plt.legend(legend, loc='upper right')
    plt.savefig("{}".format(figure_name))
    plt.show()

def main():
    # benchmarkHiddenSize([250, 100, 50, 20, 10, 5, 2, 1], "output/varHidden.png")
    benchmarkHiddenSize([150, 50, 10, 1], "output/test.png")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if(sys.argv[1] == "live"):
            live_learning(frame_delay=frame_delay)
        elif(sys.argv[1] == "record"):
            record_video(vid_out)
        elif(sys.argv[1] == "dataset"):
            dataset = Dataset.Dataset((sizeX, sizeY), frame_delay=5)
            dataset.build(path_save_dataset, from_vid=vid_in)
            print("Done ! saved on : {}".format(path_save_dataset))
        else:
            main()
    else:
        main()
