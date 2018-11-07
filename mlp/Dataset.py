import cv2
import pickle
import random
import getch
from utils import convertGray, get_center_frame, matMultiplication
from utils import print_progress

class Dataset:

    def __init__(self, data_size, with_labels=False, frame_delay=3):
        self.data_size = data_size
        self.with_labels = with_labels
        self.frame_delay = frame_delay
        self.path_to_pickle = ""
        self.data = []

    def load_dataset(self, path_to_dataset, concatenate=False):
        """

        """
        with open(path_to_dataset, "rb") as load_file:
            if(concatenate):
                self.data += pickle.load(load_file)
            else:
                self.data = pickle.load(load_file)
            load_file.close()
            print("Load : {} imgs".format(len(self.data)))
            return 0
        return 1

    def build(self, path_save_dataset, from_vid=None, size_dataset=None, concatenate=False):
        """
        Param :
            path_to_dataset : path where the dataset pickle file will be saved
            from_vid : None if the data are record in live, else path to video
            size_dataset : number of images you want to limit the dataset size
            concatenate : True if you want to get more data without loosing the
                          the previous data
        """
        if(from_vid):
            cap = cv2.VideoCapture(from_vid)
        else:
            cap = cv2.VideoCapture(2)

        if not cap.isOpened():
            print("Unable to connect to camera.")
            return 1

        ret, frame = cap.read()
        frame_gray = convertGray(frame)
        prev_gray = frame_gray

        if(not concatenate):
            self.data = []

        n_img, n = 0, 0
        while cap.isOpened():

            ret, frame = cap.read()
            frame_gray = convertGray(frame)

            if ret and n%self.frame_delay==0:
                previous = prev_gray
                prev_gray = frame_gray

                prev_face, points = get_center_frame(previous, self.data_size)
                frame_face, points = get_center_frame(frame_gray,self.data_size)

                mult_face = matMultiplication(prev_face, frame_face)

                self.data.append(mult_face)
                n_img += 1
                if(size_dataset):
                    print_progress(n_img, 0, size_dataset)
                    if(n_img == size_dataset):
                        break

                # print(n_img)
                cv2.rectangle(frame, points[0], points[1], (0,255,0), 3)
                cv2.imshow("live", frame)

            n+=1
            if(from_vid):
                # print("{}/{}".format(n, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                print_progress(n, 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1 == n):
                    break
            if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
                break

        # shuffle data
        random.shuffle(self.data)

        # save on a pickle file
        with open(path_save_dataset, "wb") as save_file:
            print("Saved : {} imgs".format(len(self.data)))
            pickle.dump(self.data, save_file)
            save_file.close()

    def build_labeled(self, path_save_dataset, delta_frame=10, concatenate=False):

        cap = cv2.VideoCapture(2)
        ret, first = cap.read()
        unknown = True
        label = 0

        while(cap.isOpened()):
            keyInt = ord(getch.getch())

            if(keyInt == 27): #echap
                break
            elif(keyInt == 122): #z
                print("Haut")
                label = 1
                unknown = False
            elif(keyInt == 115): #s
                print("Bas")
                label = 2
                unknown = False
            elif(keyInt == 100): #d
                print("Droite")
                label = 3
                unknown = False
            elif(keyInt == 113): #q
                print("Gauche")
                label = 4
                unknown = False
            elif(keyInt == 105): #i
                print("Avant")
                label = 5
                unknown = False
            elif(keyInt == 111): #o
                print("Arriere")
                label = 6
                unknown = False
            else:
                print("Unknown key")
                unknown = True

            if(not unknown):
                ret, first = cap.read()
                cv2.waitKey(delta_frame)
                ret, second = cap.read()

                prev_face, points = get_center_frame(convertGray(first), self.data_size)
                next_face, points = get_center_frame(convertGray(second),self.data_size)

                mult_face = matMultiplication(prev_face, next_face)
                if(self.with_labels):
                    self.data.append((mult_face, label))
                else:
                    self.data.append(mult_face)

                cv2.imshow("Multiplication", cv2.resize(mult_face, (300, 300)))

        print(len(self.data))


    def show_dataset(self, begin=0, end=1, delay=250):
        """Show a range of the dataset"""
        for i in range(begin, end+1):
            cv2.imshow(str(i), self.data[i])
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

# dataset = Dataset((80, 80), with_labels=True)
# dataset.build_labeled("de")
