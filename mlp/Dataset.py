import cv2
import numpy as np
import math
import pickle
import random
import datetime
import getch
from utils import convertGray, get_center_frame, matMultiplication
from utils import print_progress
import matplotlib
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self, img_size):
        self.img_size = img_size
        self.data = []
        self.X = []
        self.Y = []

    def load(self, path_to_dataset):
        """

        """
        with open(path_to_dataset, "rb") as load_file:
            print("Load from {} ...".format(path_to_dataset))
            self.data = pickle.load(load_file)

            print("Data shape : {}".format(self.data.shape))
            return 0
        return 1

    def build(self, path_save_dataset, from_vid=None):
        """
        Param :
            path_to_dataset : path where the dataset pickle file will be saved
            from_vid : None if the data are record in live, else path to video
            size_dataset : number of images you want to limit the dataset size
            concatenate : True if you want to get more data without loosing the
                          the previous data
        """
        self.data = np.asarray(self.data)
        self.data = self.data.tolist()
        if(from_vid):
            cap = cv2.VideoCapture(from_vid)
        else:
            cap = cv2.VideoCapture(2)

        if not cap.isOpened():
            print("Unable to connect to camera.")
            return 1

        ret, frame = cap.read()
        frame_gray = convertGray(frame)

        n_img = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame_gray = convertGray(frame)
            gray_face, points = get_center_frame(frame_gray, self.img_size)

            self.data.append(gray_face)

            n_img += 1

            cv2.rectangle(frame, points[0], points[1], (0,255,0), 3)
            cv2.imshow("live", frame)

            if(from_vid):
                if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1 == n_img):
                    break

            if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
                break

        self.data = np.asarray(self.data)

        # save on a pickle file
        with open(path_save_dataset, "wb") as save_file:
            print("Save in {} ...".format(path_save_dataset))
            pickle.dump(self.data, save_file)
            print("Data shape : {}".format(self.data.shape))

    def split_and_process(self, delta_time, test_ratio=0.3, valid_ratio=0.3):

        #SPLIT X AND Y WITH A DELTA OF TIME
        for i in range(delta_time, len(self.data)):
            self.X.append(self.data[i-delta_time])
            self.Y.append(self.data[i])
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y)

        #NORMALIZE DATA
        self.X -= self.X.mean(0)[None, :]
        self.X /= self.X.std(0)[None, :] + self.X.std() * 0.1
        self.X /= np.amax(self.X)
        self.Y -= self.Y.mean(0)[None, :]
        self.Y /= self.Y.std(0)[None, :] + self.Y.std() * 0.1
        self.Y /= np.amax(self.Y)

        #SHUFFLE TRAIN DATA
        R = np.random.permutation(self.X.shape[0])
        self.X = self.X[R, :]
        self.Y = self.Y[R, :]

        #SPLIT DATASET
        idx = int(test_ratio*self.X.shape[0])
        idx2 = int(valid_ratio*self.X.shape[0]) + idx
        x_test = self.X[:idx]
        y_test = self.Y[:idx]
        x_valid = self.X[idx:idx2]
        y_valid = self.Y[idx:idx2]
        x_train = self.X[idx2:]
        y_train = self.Y[idx2:]

        print("Train :\tX {}\tY {}".format(x_train.shape, y_train.shape))
        print("Test :\tX {}\tY {}".format(x_test.shape, y_test.shape))
        print("Valid :\tX {}\tY {}".format(x_valid.shape, y_valid.shape))

        return x_train, y_train, x_test, y_test, x_valid, y_valid

    def process(self, delta_time):

        #SPLIT X AND Y WITH A DELTA OF TIME
        for i in range(delta_time, len(self.data)):
            self.X.append(self.data[i-delta_time])
            self.Y.append(self.data[i])
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y)

        #NORMALIZE DATA
        self.X -= self.X.mean(0)[None, :]
        self.X /= self.X.std(0)[None, :] + self.X.std() * 0.1
        self.X /= np.amax(self.X)
        self.Y -= self.Y.mean(0)[None, :]
        self.Y /= self.Y.std(0)[None, :] + self.Y.std() * 0.1
        self.Y /= np.amax(self.Y)

        print("Size - X: {}\tY: {}".format(self.X.shape, self.Y.shape))
        return self.X, self.Y


    def show_dataset(self, begin=0, end=1, delay=250):
        """Show a range of the dataset"""
        for i in range(begin, end+1):
            cv2.imwrite('../dataset/sample/sample_{}_X.png'.format(i),
                        self.x_train[i])
            cv2.imwrite('../dataset/sample/sample_{}_Y.png'.format(i),
                        self.y_train[i])
            cv2.imshow("x_{}".format(i), self.x_train[i])
            cv2.imshow("y_{}".format(i), self.y_train[i])
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

    def plot_image(self, idx):
        fig=plt.figure(figsize=(8, 8))
        fig.add_subplot(1, 3, 1)
        plt.imshow(self.x_train[idx], cmap='gray')
        fig.add_subplot(1, 3, 2)
        plt.imshow(self.y_train[idx], cmap='gray')
        fig.add_subplot(1, 3, 3)
        plt.imshow(matMultiplication(self.x_train[idx], self.y_train[idx]),
                   cmap='gray')
        plt.show()
        now = datetime.datetime.now()
        fig.savefig('../dataset/sample/{}.png'.format(now))

    def plot_dataset(self, begin=0, end=1):
        fig=plt.figure(figsize=(8, 8))
        nb_img = end - begin +1
        nb_plot = 1
        idx = begin
        columns = 3
        rows = nb_img
        for i in range(1, nb_img+1):
            ax = fig.add_subplot(rows, columns, nb_plot)
            plt.imshow(self.x_train[idx], cmap='gray')
            ax.set_title("X")
            nb_plot += 1
            ax = fig.add_subplot(rows, columns, nb_plot)
            plt.imshow(self.y_train[idx], cmap='gray')
            ax.set_title("Y")
            nb_plot += 1
            ax = fig.add_subplot(rows, columns, nb_plot)
            plt.imshow(matMultiplication(self.x_train[idx], self.y_train[idx]),
                       cmap='gray')
            ax.set_title("X*Y")
            nb_plot += 1
            idx += 1
        plt.show()
        now = datetime.datetime.now()
        fig.savefig('../dataset/sample/{}.png'.format(now))
