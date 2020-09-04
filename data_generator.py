import os
import csv
import random
import cv2
from imageio import imread
import pandas as pd
from skimage.transform import resize

cascade_path = 'model\cv2\haarcascade_frontalface_alt2.xml'

### image size that for resizing image to only face after detection
image_size = 160

os.chdir("./CASIA-WebFace")
whole_data_path = os.getcwd()
folderList = os.listdir()


def check_if_face_exists(filepath):
    cascade = cv2.CascadeClassifier(cascade_path)
    img = imread(filepath)
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(faces) == 0:
        return False
    (x, y, w, h) = faces[0]
    try:
        cropped = img[max(y - 10 // 2, 0):y + h + 10 // 2, x - 10 // 2:x + w + 10 // 2, :]
        # cropped = img[y - 10 // 2:y + h + 10 // 2, x - 10 // 2:x + w + 10 // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
    except ValueError:
        return False
    except IndexError:
        return False
    return True


os.chdir("../")
n = 0
m = 0

### function that choose 2 random subfolders (people) for generating labeled data
def choose_random_person():
    n = random.randint(1, len(folderList) - 1)
    m = random.randint(1, len(folderList) - 1)
    first_person_root_dir = ""
    second_person_root_dir = ""
    while (m == n):
        m = random.randint(1, len(folderList))
    ## get photo file names
    for root, dirs, files in os.walk(whole_data_path + '\\' + folderList[n]):
        first_person_root_dir = root
        files_in_n = files

    for root, dirs, files in os.walk(whole_data_path + '\\' + folderList[m]):
        second_person_root_dir = root
        files_in_m = files


# writer.writerow(["first", "second", "label"])
writer_arr = []

### loop to select random photos from chooden subfolders to generate dataset for given length
### Note output will be data_length*2 length because there will be data_length positive and data_length negative examples
data_length = 3500
for i in range(0, data_length):

    if i % 10 == 0:
        print('{}/{}'.format(i, data_length))

    ##choose person
    n = 0
    m = 0
    first_person_root_dir = ""
    second_person_root_dir = ""
    files_in_m = []
    files_in_n = []

    choose_random_person()
    first_check_counter = 0
    second_check_counter = 0
    ##choose random 2 photos from n for same labeled
    first_same_in_n = random.choice(files_in_n)
    while not check_if_face_exists(first_person_root_dir + "\\" + first_same_in_n):
        first_same_in_n = random.choice(files_in_n)
    second_same_in_n = random.choice(files_in_n)
    while second_same_in_n == first_same_in_n or (
    not check_if_face_exists(first_person_root_dir + "\\" + second_same_in_n)):
        second_same_in_n = random.choice(files_in_n)
    ##choose random photo from m for different labeled
    random_from_m = random.choice(files_in_m)
    while not check_if_face_exists(second_person_root_dir + "\\" + random_from_m):
        random_from_m = random.choice(files_in_m)
    writer_arr.append([whole_data_path + '\\' + folderList[n] + '\\' + first_same_in_n,
                       whole_data_path + '\\' + folderList[n] + '\\' + second_same_in_n, "1"])

    writer_arr.append([whole_data_path + '\\' + folderList[n] + '\\' + first_same_in_n,
                       whole_data_path + '\\' + folderList[m] + '\\' + random_from_m, "0"])
### write collected image pairs to csv
mydf = pd.DataFrame(writer_arr, columns=['image1', 'image2', 'label'])
mydf.to_csv('formatted-7000-data.csv', sep=',')
