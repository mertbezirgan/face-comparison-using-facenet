import os
import csv
import random
import cv2
from imageio import imread
import pandas as pd
from skimage.transform import resize
import numpy as np

labeled_data = pd.read_csv("./formatted-7000-data.csv")

image_size = 160

os.chdir("./CASIA-WebFace")
whole_data_path = os.getcwd()
folderList = os.listdir()

cascade_path = '..\model\cv2\haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascade_path)
def check_if_face_exists(filepath):

    img = imread(filepath)
    # img = imread(filepath)
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(faces) == 0:
        return False
    (x, y, w, h) = faces[0]
    cropped = img[max(y - 10 // 2, 0):y + h + 10 // 2, x - 10 // 2:x + w + 10 // 2, :]
    aligned = resize(cropped, (image_size, image_size), mode='reflect')
    try:
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
    except ValueError:
        return False
    except IndexError:
        return False
    return True



real_results = np.array(labeled_data["label"])
image1_paths = np.array(labeled_data["image1"])
image2_paths = np.array(labeled_data["image2"])
check = []
for i in range(5067, 5068):
    print(check_if_face_exists(image2_paths[i]), i)
