import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def convertGray(image):
    return np.dot(image[...,:3], [0.299/256, 0.587/256, 0.114/256])

def get_center_frame(img, out_size, img_scale=0.5):
    w_img = img.shape[1]
    h_img = img.shape[0]

    x_TL = int((w_img/2)-(img_scale*w_img/2))
    y_TL = int((h_img/2)-(img_scale*w_img/2))

    x_BR = int((w_img/2)+(img_scale*w_img/2))
    y_BR = int((h_img/2)+(img_scale*w_img/2))

    center = img[y_TL:y_BR, x_TL:x_BR]
    ctr_resized = cv2.resize(center, out_size)

    return ctr_resized, ((x_TL,y_TL), (x_BR,y_BR))

def matMultiplication(image1, image2):
    height, width = image1.shape[0], image1.shape[1]
    retour = np.zeros((height, width), np.float)

    retour = image1*image2 - image1
    # from [-1, 0] to [-1, 1]
    # retour += 1
    # retour *= 2
    # retour -= 1

    return retour

def print_progress(current, min, max, size=80):
    position = current-min
    max_borne = max - min
    done_size = int(position * size / max_borne)
    line = "|" + "="*(done_size-1) + ">" + " "*(size-done_size-1) + "|"
    print(line, end='\r')
    if(done_size == size-1):
        print("")

def plot_error_evolution(errs, figure_name):
    fig, ax = plt.subplots()
    ax.plot(errs)

    ax.set(xlabel='Iterations', ylabel='Erreur quadratique',
           title='Evolution de l\'erreur quadratique avec le temps')
    ax.grid()

    fig.savefig(figure_name)
    plt.show()

def record_video(file_name, frame_delay=1, vid_size=None):

    cap = cv2.VideoCapture(2)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, 20.0, (640,480))

    nb_img = 1
    n=0
    print("Save in {}".format(file_name))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret and n%frame_delay==0:
            print(nb_img)
            out.write(frame)
            nb_img += 1
            center, points = get_center_frame(frame, (40, 40))
            cv2.rectangle(frame, points[0], points[1], (0,255,0), 3)
            cv2.imshow('frame',frame)

            if(vid_size):
                if nb_img >= vid_size:
                    break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        n += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    record_video("../dataset/louis2.avi")
